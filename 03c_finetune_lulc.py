#!/usr/bin/env python3
"""
Step 3c: Re-run the California fine-tuning with land use added as a covariate.

Leave-one-out showed the model resolves species means but almost nothing within a
species (13% of within-species variance for He, 1% for pi, median within-species
r = 0.01), because the climate covariates carry little within-species signal —
gradient boosting tops out near R² = 0.05 on the same data. NLCD land-use
pressure correlates with within-species anomalies at median |r| = 0.26, an order
of magnitude more, so it is the most promising extra explanatory variable.

Phase 1 (global) is untouched: NLCD is US-only and cannot be extracted at the
19,163 global populations. Instead the pre-trained encoder is WIDENED — the new
land-use inputs enter through zero-initialised weights, so the model starts out
computing exactly what the pre-trained one did and can only improve by learning
to use them — and only the California fine-tuning is re-run.

Targets stay in their native units: raw He (identity link) and pi (log link),
per-metric standardized inside the processor and inverted on prediction.

Outputs:
  models/cnp_california_lulc.pt
  models/feature_processor_lulc.pkl
  data/processed/loo_lulc_comparison.csv
"""

import pickle, sys, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
FeatureProcessor = _m.FeatureProcessor
ConditionalNeuralProcess = _m.ConditionalNeuralProcess
NUMERIC_FEATURES = _m.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _m.CATEGORICAL_FEATURES
REPR_DIM = _m.REPR_DIM
DEVICE = _m.DEVICE

import lulc_model
from lulc_model import LulcProcessor, LULC_FEATURES, widen_encoder

P2_EPOCHS = 80
P2_EPISODES = 100


def build_data():
    cali = pd.read_csv(PROC / "california_env_lulc.csv")
    cali = _m.filter_ca_species(cali)
    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    # Global records carry no NLCD; zeros plus lulc_known=0 tell the model so.
    for f in LULC_FEATURES:
        global_df[f] = 0.0
    for col in CATEGORICAL_FEATURES:
        for df in (cali, global_df):
            if col not in df.columns:
                df[col] = "unknown"
    return cali, global_df


def main():
    print("=" * 62)
    print("PHASE 2 RE-RUN: California fine-tuning with land use")
    print("=" * 62)

    cali, global_df = build_data()
    print(f"California: {len(cali)} records, {cali['species'].nunique()} species")
    print(f"  with NLCD coverage: {int(cali['lulc_known'].sum())}")
    print(f"Extra covariates: {', '.join(LULC_FEATURES)}")

    with open(MODEL_DIR / "feature_processor.pkl", "rb") as f:
        base_proc = pickle.load(f)
    proc = LulcProcessor(base_proc).fit_lulc(cali[cali["lulc_known"] == 1])

    n_base = len(NUMERIC_FEATURES)
    n_new = len(LULC_FEATURES)

    model = ConditionalNeuralProcess(
        n_numeric=n_base + n_new,
        cat_cardinalities=base_proc.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2,
        n_species=base_proc.n_species,
        repr_dim=REPR_DIM,
        dropout=0.1,
    ).to(DEVICE)

    pre = torch.load(MODEL_DIR / "cnp_california_finetuned.pt",
                     map_location="cpu", weights_only=True)
    pre = widen_encoder(pre, n_base, n_new)
    missing, unexpected = model.load_state_dict(pre, strict=False)
    print(f"\nStarted from cnp_california_finetuned.pt, encoder widened "
          f"{n_base} -> {n_base + n_new} numeric inputs")
    if missing or unexpected:
        print(f"  (missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Fine-tune ──
    within = _m.WithinSpeciesCNPDataset(cali, proc, min_pops=2)
    mixed = _m.MixedContextCNPDataset(cali, global_df, proc, min_pops=2)

    opt = torch.optim.AdamW([
        {"params": model.feature_encoder.mlp.parameters(), "lr": 1e-4},
        {"params": model.feature_encoder.film_gamma.parameters(), "lr": 5e-4},
        {"params": model.feature_encoder.film_beta.parameters(), "lr": 5e-4},
        {"params": model.feature_encoder.metric_embed.parameters(), "lr": 5e-4},
        {"params": model.feature_encoder.species_embed.parameters(), "lr": 2e-4},
        {"params": model.feature_encoder.cat_embeddings.parameters(), "lr": 2e-5},
        {"params": model.obs_encoder.parameters(), "lr": 1e-4},
        {"params": model.decoder.parameters(), "lr": 2e-4},
        {"params": model.mu_head.parameters(), "lr": 2e-4},
        {"params": model.log_sigma_head.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P2_EPOCHS)

    print(f"\nFine-tuning {P2_EPOCHS} epochs x {P2_EPISODES} episodes")
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(1, P2_EPOCHS + 1):
        loss = _m.train_epoch(model, [within, mixed], opt,
                              n_episodes=P2_EPISODES, mix_weights=[0.5, 0.5])
        met = _m.evaluate(model, within, n_episodes=100)
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}: loss={loss:.4f}  RMSE={met['rmse']:.4f}  "
                  f"R²={met['r2']:.4f}")
        if met["rmse"] < best:
            best, patience = met["rmse"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "cnp_california_lulc.pt")
    with open(MODEL_DIR / "feature_processor_lulc.pkl", "wb") as f:
        pickle.dump(proc, f)
    print(f"\nSaved: models/cnp_california_lulc.pt, models/feature_processor_lulc.pkl")

    # ── Does it resolve more within-species variation? ──
    print("\nLeave-one-out, within-species skill:")
    _p = __import__("04_predict_california")
    base_model, base_processor = _p.load_model_and_processor()

    rows = []
    for sp, g in cali.groupby("species"):
        if len(g) < 6:
            continue
        for tag, mdl, pr in [("baseline", base_model, base_processor),
                             ("with_lulc", model, proc)]:
            mu, sig, y = _p.leave_one_out_eval(mdl, g, pr)
            if len(mu) == 0:
                continue
            for a, b in zip(y, mu):
                rows.append({"species": sp, "model": tag, "obs": a, "pred": b})
    d = pd.DataFrame(rows)
    d.to_csv(PROC / "loo_lulc_comparison.csv", index=False)

    print(f"\n  {'species':26s} {'n':>3s} {'r baseline':>11s} {'r +LULC':>9s}")
    for sp, g in d.groupby("species"):
        vals = {}
        for tag, gg in g.groupby("model"):
            vals[tag] = (np.corrcoef(gg.obs, gg.pred)[0, 1]
                         if gg.pred.std() > 1e-12 else np.nan)
        print(f"  {sp[:26]:26s} {len(g)//2:3d} {vals.get('baseline', np.nan):11.2f} "
              f"{vals.get('with_lulc', np.nan):9.2f}")

    for tag, g in d.groupby("model"):
        wv = g.groupby("species").apply(
            lambda s: pd.Series({"vo": ((s.obs - s.obs.mean()) ** 2).mean(),
                                 "vp": ((s.pred - s.pred.mean()) ** 2).mean()}))
        r = g.groupby("species").apply(
            lambda s: np.corrcoef(s.obs, s.pred)[0, 1] if s.pred.std() > 1e-12 else np.nan)
        print(f"\n  {tag}: within-species predicted variance = "
              f"{100 * wv.vp.sum() / wv.vo.sum():.0f}% of observed, median r = {r.median():+.2f}")


if __name__ == "__main__":
    main()

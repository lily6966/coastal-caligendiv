#!/usr/bin/env python3
"""
Step 3b: Species-level k-fold cross-validation of the California fine-tuning.

Why: the single 30% species hold-out in 03_train_model.py gives noisy metrics
(only ~2-3 test species per metric). This script produces manuscript-grade,
raw-unit metrics with exact (per-metric, log-link) credible intervals by:

  - Loading the existing global-pretrained checkpoint (Phase 1 is independent of
    the California split, so it is reused, not retrained).
  - Stratified k-fold over the 16 California species (folds balanced by metric).
  - For each fold: fine-tune a fresh copy from the pretrained weights on the
    training species for a FIXED number of epochs (no early stopping, no peeking
    at held-out data), then predict every population of each held-out species by
    within-species leave-one-out (context = the species' other populations).
  - Pool all out-of-fold population predictions and report RMSE, R^2 and 95%/68%
    interval coverage in RAW units, separately for He and pi.

Outputs: data/processed/crossval_metrics.csv and a console summary.
"""

import sys, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
import os
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
ConditionalNeuralProcess = _m.ConditionalNeuralProcess
FeatureProcessor = _m.FeatureProcessor
WithinSpeciesCNPDataset = _m.WithinSpeciesCNPDataset
MixedContextCNPDataset = _m.MixedContextCNPDataset
train_epoch = _m.train_epoch
filter_ca_species = _m.filter_ca_species
NUMERIC_FEATURES = _m.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _m.CATEGORICAL_FEATURES
REPR_DIM = _m.REPR_DIM
DEVICE = _m.DEVICE

# Evaluate the final 1 km-climate + NLCD land-use model.
import lulc_model
from lulc_model import LulcProcessor, LULC_FEATURES, widen_encoder
USE_LULC = True
CALI_FILE = "california_env_lulc.csv" if USE_LULC else "california_env.csv"

K_FOLDS = 5
P2_EPOCHS = 80
P2_EPISODES = 100
SEED = 42


def build_finetuned(pretrained_state, processor, cali_train, global_df):
    """Fresh fine-tune from the global-pretrained weights on the training species.
    For the LULC model the encoder is zero-widened for the extra land-use inputs,
    exactly as in 03c, so held-out species never influence the starting point."""
    n_base = len(NUMERIC_FEATURES)
    n_new = len(LULC_FEATURES) if USE_LULC else 0
    model = ConditionalNeuralProcess(
        n_numeric=n_base + n_new,
        cat_cardinalities=processor.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2, n_species=processor.n_species,
        repr_dim=REPR_DIM, dropout=0.1,
    ).to(DEVICE)
    pre = widen_encoder(pretrained_state, n_base, n_new) if USE_LULC else pretrained_state
    ft_state = model.state_dict()
    for k in pre:
        if k in ft_state and pre[k].shape == ft_state[k].shape:
            ft_state[k] = pre[k]
    model.load_state_dict(ft_state)

    within = WithinSpeciesCNPDataset(cali_train, processor, min_pops=2)
    mixed = MixedContextCNPDataset(cali_train, global_df, processor, min_pops=2)
    datasets = [d for d in (within, mixed) if len(d.groups) > 0]
    if not datasets:
        return None

    opt = torch.optim.AdamW([
        {"params": model.feature_encoder.mlp.parameters(), "lr": 2e-5},
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

    w = [0.5] * len(datasets)
    w = [x / sum(w) for x in w]
    for _ in range(P2_EPOCHS):
        train_epoch(model, datasets, opt, n_episodes=P2_EPISODES, mix_weights=w)
        sched.step()
    return model


def loo_predict_species(model, sp_df, processor):
    """Within-species leave-one-out: standardized z_mu, z_sigma per population."""
    model.eval()
    n = len(sp_df)
    out = []
    for i in range(n):
        ctx = sp_df.drop(sp_df.index[i])
        tgt = sp_df.iloc[[i]]
        fc, ft = processor.transform(ctx), processor.transform(tgt)
        with torch.no_grad():
            ctx_repr = model.encode_context(
                torch.tensor(fc["numeric"]).to(DEVICE),
                {k: torch.tensor(v).to(DEVICE) for k, v in fc["categoricals"].items()},
                torch.tensor(fc["metric_type"]).to(DEVICE),
                torch.tensor(fc["species_id"]).to(DEVICE),
                torch.tensor(fc["target"]).to(DEVICE),
            )
            tf = model.feature_encoder(
                torch.tensor(ft["numeric"]).to(DEVICE),
                {k: torch.tensor(v).to(DEVICE) for k, v in ft["categoricals"].items()},
                torch.tensor(ft["metric_type"]).to(DEVICE),
                torch.tensor(ft["species_id"]).to(DEVICE),
            )
            z_mu, z_sigma = model.decode(ctx_repr, tf)
        out.append((float(z_mu.cpu().numpy()[0]), float(z_sigma.cpu().numpy()[0])))
    return out


def main():
    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    cali_df = pd.read_csv(PROC / CALI_FILE)
    cali_df = filter_ca_species(cali_df)
    for col in CATEGORICAL_FEATURES:
        if col not in cali_df.columns:
            cali_df[col] = "unknown"
        if col not in global_df.columns:
            global_df[col] = "unknown"
    if USE_LULC:  # global records carry no NLCD; zeros + lulc_known=0 tell the model so
        for f in LULC_FEATURES:
            global_df[f] = 0.0

    # Base processor: numeric scaler + He target stats on global only, vocab on
    # combined, pi target stats on California (matches production fit).
    base_proc = FeatureProcessor()
    base_proc.fit(global_df, cali_df)
    pretrained = torch.load(MODEL_DIR / "cnp_global_pretrained.pt",
                            weights_only=True, map_location=DEVICE)

    # Stratified k-fold species assignment (balanced by metric)
    sp_metric = cali_df.groupby("species")["metric_type"].first()
    rng = np.random.RandomState(SEED)
    fold_of = {}
    for mt in ["He", "pi"]:
        sps = list(sp_metric[sp_metric == mt].index)
        rng.shuffle(sps)
        for j, sp in enumerate(sps):
            fold_of[sp] = j % K_FOLDS
    print(f"Species: {len(sp_metric)} ({(sp_metric=='He').sum()} He, "
          f"{(sp_metric=='pi').sum()} pi) across {K_FOLDS} folds")

    rows = []  # pooled out-of-fold population predictions
    for fold in range(K_FOLDS):
        test_sp = [s for s, f in fold_of.items() if f == fold]
        train_sp = [s for s, f in fold_of.items() if f != fold]
        cali_train = cali_df[cali_df["species"].isin(train_sp)]
        print(f"\nFold {fold+1}/{K_FOLDS}: {len(train_sp)} train / {len(test_sp)} held-out species")
        # LULC scaler fit on training species only (no held-out leakage into scaling)
        if USE_LULC:
            proc = LulcProcessor(base_proc).fit_lulc(cali_train[cali_train["lulc_known"] == 1])
        else:
            proc = base_proc
        model = build_finetuned(pretrained, proc, cali_train, global_df)
        if model is None:
            continue
        for sp in test_sp:
            sp_df = cali_df[cali_df["species"] == sp]
            if len(sp_df) < 2:
                continue
            mt = sp_df["metric_type"].iloc[0]
            preds = loo_predict_species(model, sp_df, proc)
            for (z_mu, z_sigma), (_, r) in zip(preds, sp_df.iterrows()):
                rows.append({"species": sp, "metric_type": mt,
                             "y": float(r["gen_div"]), "z_mu": z_mu, "z_sigma": z_sigma})
        print(f"  done ({len([r for r in rows if r['species'] in test_sp])} pooled preds so far)")

    res = pd.DataFrame(rows)
    # Back-transform to raw units + exact per-metric bounds (target link lives on base)
    res["pred_mu"] = base_proc.inverse_target(res["z_mu"].values, res["metric_type"].values)
    lo95, hi95 = base_proc.inverse_bounds(res["z_mu"].values, res["z_sigma"].values,
                                          res["metric_type"].values, 1.96)
    lo68, hi68 = base_proc.inverse_bounds(res["z_mu"].values, res["z_sigma"].values,
                                          res["metric_type"].values, 1.0)
    res["lo95"], res["hi95"], res["lo68"], res["hi68"] = lo95, hi95, lo68, hi68
    res.to_csv(PROC / "crossval_predictions.csv", index=False)

    print("\n" + "=" * 64)
    print(f"{K_FOLDS}-FOLD SPECIES-LEVEL CROSS-VALIDATION (raw units, exact intervals)")
    print("=" * 64)
    summary = []
    for mt in ["He", "pi", "all"]:
        d = res if mt == "all" else res[res["metric_type"] == mt]
        if len(d) == 0:
            continue
        y, mu = d["y"].values, d["pred_mu"].values
        rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
        ss_res = np.sum((y - mu) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        cov95 = float(np.mean((y >= d["lo95"]) & (y <= d["hi95"])))
        cov68 = float(np.mean((y >= d["lo68"]) & (y <= d["hi68"])))
        neg = float(np.mean(d["lo95"] < 0))
        pearson = float(np.corrcoef(y, mu)[0, 1]) if np.std(mu) > 1e-12 else float("nan")
        n_sp = d["species"].nunique()
        summary.append({"metric": mt, "n_species": n_sp, "n_pops": len(d),
                        "RMSE": rmse, "R2": r2, "pearson_r": pearson,
                        "cov95": cov95, "cov68": cov68, "frac_CI_below0": neg})
        print(f"  {mt:3s}: n_sp={n_sp:2d} n_pops={len(d):4d}  RMSE={rmse:.5f}  "
              f"R2={r2:.3f}  r={pearson:.3f}  cov95={cov95:.3f}  cov68={cov68:.3f}  CI<0={neg*100:.0f}%")
    pd.DataFrame(summary).to_csv(PROC / "crossval_metrics.csv", index=False)
    print(f"\nSaved: {PROC/'crossval_metrics.csv'} and {PROC/'crossval_predictions.csv'}")


if __name__ == "__main__":
    main()

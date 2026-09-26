#!/usr/bin/env python3
"""Global observed-vs-predicted He from the global-pretrained CNP.

For each global species with >=4 populations, a random 70% forms the context and
the held-out ~30% are predicted (pooled across species). Reports RMSE / R^2 in raw
He units and saves figures/global_obs_vs_pred.{pdf,png}.
"""
import sys, pickle
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed")); MODEL_DIR = ROOT / "models"
FIG = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
FeatureProcessor = _m.FeatureProcessor  # bind for pickle
DEVICE = _m.DEVICE
rng = np.random.RandomState(0)

with open(MODEL_DIR / "feature_processor.pkl", "rb") as f:
    proc = pickle.load(f)
model = _m.ConditionalNeuralProcess(
    n_numeric=len(_m.NUMERIC_FEATURES), cat_cardinalities=proc.cat_cardinalities,
    cat_embed_dims=_m.CATEGORICAL_FEATURES, n_metric_types=2,
    n_species=proc.n_species, repr_dim=_m.REPR_DIM, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(MODEL_DIR / "cnp_global_pretrained.pt",
                                 weights_only=True, map_location=DEVICE))
model.eval()

g = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
for col in _m.CATEGORICAL_FEATURES:
    if col not in g.columns:
        g[col] = "unknown"

def enc(df):
    f = proc.transform(df)
    return (torch.tensor(f["numeric"]).to(DEVICE),
            {k: torch.tensor(v).to(DEVICE) for k, v in f["categoricals"].items()},
            torch.tensor(f["metric_type"]).to(DEVICE),
            torch.tensor(f["species_id"]).to(DEVICE),
            torch.tensor(f["target"]).to(DEVICE))

ys, mus = [], []
for sp, d in g.groupby("species"):
    n = len(d)
    if n < 4:
        continue
    perm = rng.permutation(n)
    n_ctx = max(3, int(n * 0.7))
    ctx, tgt = d.iloc[perm[:n_ctx]], d.iloc[perm[n_ctx:]]
    if len(tgt) == 0:
        continue
    cn, cc, cm, cs, cy = enc(ctx)
    tn, tc, tm, ts, _ = enc(tgt)
    with torch.no_grad():
        cr = model.encode_context(cn, cc, cm, cs, cy)
        tf = model.feature_encoder(tn, tc, tm, ts)
        zmu, _ = model.decode(cr, tf)
    mu = proc.inverse_target(zmu.cpu().numpy(), np.array(["He"] * len(tgt)))
    ys.extend(tgt["gen_div"].values); mus.extend(mu)

ys, mus = np.array(ys), np.array(mus)
rmse = float(np.sqrt(np.mean((mus - ys) ** 2)))
r2 = float(1 - np.sum((ys - mus) ** 2) / np.sum((ys - ys.mean()) ** 2))
print(f"Global He obs-vs-pred: n={len(ys)} pops  RMSE={rmse:.4f}  R2={r2:.3f}")

fig, ax = plt.subplots(figsize=(6.2, 6))
hb = ax.hexbin(ys, mus, gridsize=45, bins="log", cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax, label="log10(count)", shrink=0.85)
lim = [min(ys.min(), mus.min()), max(ys.max(), mus.max())]
ax.plot(lim, lim, "r--", lw=1.5, label="1:1")
ax.set_xlabel("Observed He"); ax.set_ylabel("Predicted He")
ax.set_title(f"Global held-out predictions (CNP pre-training)\n"
             f"n={len(ys)} populations across {g['species'].nunique()} species  |  "
             f"R²={r2:.2f}, RMSE={rmse:.3f}")
ax.legend(loc="upper left"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
plt.tight_layout()
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    plt.savefig(FIG / f"global_obs_vs_pred.{ext}", bbox_inches="tight", **kw)
print(f"Saved: {FIG/'global_obs_vs_pred.pdf'}")

#!/usr/bin/env python3
"""Scatter of raw predicted genetic diversity (before normalization), CNP vs IDW,
with dots colored by species. Two panels (He, pi)."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
import idw_comparison as idw

obs = _m.filter_ca_species(pd.read_csv(PROC / "california_env_lulc.csv"), verbose=False)
v = pd.read_csv(PROC / "vulnerability_scores.csv").reset_index(drop=True)

# IDW raw diversity at each grid point from that species' observed populations
pred = np.full(len(v), np.nan)
for sp, g in v.groupby("species"):
    o = obs[obs["species"] == sp]
    pred[g.index.values] = idw.idw_predict(g["Latitude"].values, g["Longitude"].values,
                                           o["Latitude"].values, o["Longitude"].values,
                                           o["gen_div"].values)
v["idw_mu"] = pred

# ── Figure 1: species-colored scatter (idw_vs_cnp_diversity) ──
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
for ax, mt in zip(axes, ["He", "pi"]):
    lab = "π" if mt == "pi" else "He"
    d = v[v["metric_type"] == mt].dropna(subset=["pred_mu", "idw_mu"])
    species = sorted(d["species"].unique())
    cmap = plt.get_cmap("tab10" if len(species) <= 10 else "tab20")
    for i, sp in enumerate(species):
        s = d[d["species"] == sp]
        ax.scatter(s["idw_mu"], s["pred_mu"], s=10, alpha=0.55,
                   color=cmap(i % cmap.N), label=sp, edgecolors="none")
    lim = [min(d["idw_mu"].min(), d["pred_mu"].min()),
           max(d["idw_mu"].max(), d["pred_mu"].max())]
    ax.plot(lim, lim, "k--", lw=1.2, label="1:1")
    r = np.corrcoef(d["pred_mu"], d["idw_mu"])[0, 1]
    ax.set_xlabel(f"IDW predicted {lab}"); ax.set_ylabel(f"CNP predicted {lab}")
    ax.set_title(f"{'Nucleotide diversity (π)' if mt=='pi' else 'Expected heterozygosity (He)'}"
                 f"   r={r:.2f}")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9)
plt.suptitle("Raw predicted genetic diversity across the coastal grid: CNP vs IDW "
             "(pre-normalization), colored by species", fontsize=12)
plt.tight_layout()
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    plt.savefig(FIG / f"idw_vs_cnp_diversity.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved: figures/idw_vs_cnp_diversity.{pdf,png}")

# ── Figure 2: frequency distribution comparison (separate file, no overwrite) ──
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, mt in zip(axes, ["He", "pi"]):
    lab = "π" if mt == "pi" else "He"
    d = v[v["metric_type"] == mt].dropna(subset=["pred_mu", "idw_mu"])
    o = obs[obs["metric_type"] == mt]["gen_div"].dropna()
    allv = np.concatenate([d["pred_mu"].values, d["idw_mu"].values, o.values])
    bins = np.linspace(np.nanmin(allv), np.nanmax(allv), 40)
    ax.hist(o, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color="0.4", label=f"Observed (n={len(o)})")
    ax.hist(d["pred_mu"], bins=bins, density=True, histtype="step", lw=2.0,
            color="#1f77b4", label=f"CNP predicted (n={len(d)})")
    ax.hist(d["idw_mu"], bins=bins, density=True, histtype="step", lw=2.0,
            color="#d62728", label=f"IDW predicted (n={len(d)})")
    ax.set_xlabel(f"{lab}"); ax.set_ylabel("Density")
    ax.set_title(f"Frequency distribution — "
                 f"{'Nucleotide diversity (π)' if mt=='pi' else 'Expected heterozygosity (He)'}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.suptitle("Predicted genetic-diversity frequency distributions: CNP vs IDW vs observed "
             "(pre-normalization)", fontsize=12)
plt.tight_layout()
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    plt.savefig(FIG / f"idw_vs_cnp_diversity_distribution.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved: figures/idw_vs_cnp_diversity_distribution.{pdf,png}")

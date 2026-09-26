#!/usr/bin/env python3
"""Present-day 'Vulnerability Components' panel — single, consistent epoch.

Replaces the mixed-epoch top-left panel of the combined figure. Present-day
diversity is FIXED (not projected), so the diversity-deficit curve carries NO
between-GCM band; only Δ-climate exposure and the total vulnerability vary across
GCMs (exposure is the coming present→2100 change and differs by model).

  V = (1 - D_within) * total_stress + 0.3 σ,
  total_stress = 0.75 Δ-climate exposure + 0.25 LULC pressure.

Output: figures/vuln_components_presentday.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
GCMS = ["BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR", "MIROC6"]

df = pd.read_csv(PROC / "future_perGCM_diversity_exposure.csv")
D = df["diversity_norm_within"].values
sig = df["sigma_norm"].values
# per-population vulnerability, ensemble and per GCM
V_ens = (1 - D) * df["total_stress_future"].values + 0.3 * sig
V_g = {g: (1 - D) * df[f"total_stress_{g}"].values + 0.3 * sig for g in GCMS}

edges = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 76)   # 75 bins
ctr = (edges[:-1] + edges[1:]) / 2
rows = []
for i in range(len(edges) - 1):
    m = ((df["Latitude"] >= edges[i]) & (df["Latitude"] < edges[i + 1])).values
    if m.sum() == 0:
        continue
    exp_g = np.array([df[f"exposure_{g}"].values[m].mean() for g in GCMS])
    v_g = np.array([V_g[g][m].mean() for g in GCMS])
    rows.append({
        "lat": ctr[i],
        "div_deficit": 1 - D[m].mean(),                  # fixed (no GCM band)
        "exposure": df["climate_exposure"].values[m].mean(),
        "exposure_sd": exp_g.std(),
        "uncertainty": sig[m].mean(),
        "vuln": V_ens[m].mean(),
        "vuln_lo": v_g.min(), "vuln_hi": v_g.max(),
    })
e = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(e["lat"], e["div_deficit"], "b-o", ms=3, lw=2, alpha=0.75,
        label="Diversity deficit (1 - div_norm), present-day")

ax.plot(e["lat"], e["exposure"], "r-s", ms=3, lw=2, alpha=0.75,
        label="Δ-climate exposure (ensemble mean)")
ax.fill_between(e["lat"], e["exposure"] - e["exposure_sd"], e["exposure"] + e["exposure_sd"],
                color="red", alpha=0.15, label="Δ-exposure inter-model ±1 SD")

ax.plot(e["lat"], e["uncertainty"], "gray", ls="--", lw=1.5, alpha=0.6,
        label="Model uncertainty")

ax.plot(e["lat"], e["vuln"], "k-", lw=2.5, alpha=0.9, label="Vulnerability (ensemble mean)")
ax.fill_between(e["lat"], e["vuln_lo"], e["vuln_hi"], color="black", alpha=0.12,
                label="Vulnerability (GCM range)")

ax.set_xlim(e["lat"].min(), e["lat"].max()); ax.set_ylim(0, 1)
ax.set_xlabel("Latitude", fontsize=12); ax.set_ylabel("Score (0-1)", fontsize=12)
ax.set_title("Vulnerability Components (present-day)\n"
             "diversity fixed; Δ-climate exposure & vulnerability vary by GCM",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", frameon=True)
ax.grid(True, alpha=0.3)

for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig.savefig(FIG / f"vuln_components_presentday.{ext}", bbox_inches="tight", **kw)
plt.close()
print("Saved: figures/vuln_components_presentday.{pdf,png}")
print(f"  vulnerability: south≈{e['vuln'].iloc[:5].mean():.3f}  north≈{e['vuln'].iloc[-5:].mean():.3f}")
print(f"  mean GCM range width: {(e['vuln_hi']-e['vuln_lo']).mean():.3f}")

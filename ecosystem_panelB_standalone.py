#!/usr/bin/env python3
"""Faithful standalone of the 'Resilience Class Composition Along Coast' panel
(Panel B of figures/ecosystem_resilience.pdf).

Reproduces it exactly: it plots the present-day resilience_class stored by
05_resilience_assessment.py (classify_resilience on within-species diversity +
ensemble-mean total stress + model uncertainty), POINT-weighted (unit =
species×location predictions), in 75 latitude bins, stacked Critical→Latent→
At Risk→Resilient, with the number of species as a black line.

Output: figures/ecosystem_panelB_standalone.{pdf,png}
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

df = pd.read_csv(PROC / "vulnerability_scores.csv")   # has present-day resilience_class

edges = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 76)   # 75 bins, as in 05
ctr = (edges[:-1] + edges[1:]) / 2
rows = []
for i in range(len(edges) - 1):
    pts = df[(df["Latitude"] >= edges[i]) & (df["Latitude"] < edges[i + 1])]
    if len(pts) == 0:
        continue
    n = len(pts)
    rows.append({
        "lat": ctr[i], "n_species": pts["species"].nunique(),
        "pct_critical": (pts["resilience_class"] == "Critical").sum() / n,
        "pct_latent":   (pts["resilience_class"] == "Latent Vulnerability").sum() / n,
        "pct_at_risk":  (pts["resilience_class"] == "At Risk").sum() / n,
        "pct_resilient":(pts["resilience_class"] == "Resilient").sum() / n,
    })
e = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11, 7))
ax.fill_between(e["lat"], 0, e["pct_critical"], color="#e74c3c", alpha=0.85, label="Critical")
ax.fill_between(e["lat"], e["pct_critical"], e["pct_critical"] + e["pct_latent"],
                color="#e67e22", alpha=0.85, label="Latent Vulnerability")
ax.fill_between(e["lat"], e["pct_critical"] + e["pct_latent"],
                e["pct_critical"] + e["pct_latent"] + e["pct_at_risk"],
                color="#f1c40f", alpha=0.85, label="At Risk")
ax.fill_between(e["lat"], e["pct_critical"] + e["pct_latent"] + e["pct_at_risk"], 1.0,
                color="#2ecc71", alpha=0.85, label="Resilient")

axc = ax.twinx()
axc.plot(e["lat"], e["n_species"], "k-", lw=2, alpha=0.7)
axc.set_ylabel("Number of species", fontsize=12)

ax.set_xlim(e["lat"].min(), e["lat"].max()); ax.set_ylim(0, 1)
ax.set_xlabel("Latitude", fontsize=12)
ax.set_ylabel("Proportion of species", fontsize=12)
ax.set_title("Resilience Class Composition Along Coast\n"
             "(point-weighted; present-day classification from step 05)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.2, axis="x")

for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig.savefig(FIG / f"ecosystem_panelB_standalone.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("Saved: figures/ecosystem_panelB_standalone.{pdf,png}")

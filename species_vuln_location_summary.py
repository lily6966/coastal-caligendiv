#!/usr/bin/env python3
"""Location summary: proportion of SPECIES in each vulnerability level by latitude.

Species-equal (not point-weighted) view, quantified per GCM, using MEDIANS.

  - Diversity: each species' MEDIAN diversity, converted to a within-metric
    rank-percentile (0 = lowest-diversity species of that metric, 1 = highest).
    He and pi are ranked separately because their magnitudes are not comparable.
  - Stress: present-day total stress (Δ-climate exposure + LULC). Exposure is the
    projected present→2100 change and varies by GCM. Per species per band we take
    the median stress over its grid points for each GCM. The stacked proportions
    use the median-across-GCM stress (one classification per species, so classes
    sum to 1); the between-GCM spread is shown as the min–max range of the
    vulnerable fraction.

Outputs:
  figures/species_vuln_location_summary.{pdf,png}
  data/processed/species_vuln_location_summary.csv      (per-band proportions + GCM range)
  data/processed/species_diversity_levels.csv           (per-species median diversity + rank)
"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
_r = __import__("05_resilience_assessment")
classify_resilience = _r.classify_resilience

GCMS = ["BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR", "MIROC6"]
CLASSES = ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]
CLASS_COL = {"Resilient": "#2ecc71", "At Risk": "#f1c40f",
             "Latent Vulnerability": "#e67e22", "Critical": "#e74c3c"}

df = pd.read_csv(PROC / "future_perGCM_diversity_exposure.csv")

# ── Species MEDIAN diversity → within-metric RANK-PERCENTILE ──
sp = df.groupby("species").agg(metric=("metric_type", "first"),
                               med_div=("diversity_norm", "median"),
                               sigma=("sigma_norm", "median")).reset_index()
sp["div_level"] = np.nan
for mt, g in sp.groupby("metric"):
    r = g["med_div"].rank(method="average")               # 1..n within metric
    sp.loc[g.index, "div_level"] = (r - 1) / (len(g) - 1) if len(g) > 1 else 0.5
sp = sp.set_index("species")
sp.reset_index().to_csv(PROC / "species_diversity_levels.csv", index=False)
print("Per-species median diversity (within-metric rank-percentile):")
print(sp.sort_values("div_level")[["metric", "med_div", "div_level"]].round(3).to_string())

def _classify(div, stress, sigma):
    return classify_resilience(np.array([div]), np.array([stress]), np.array([sigma]))[0]

DIV_MAP = sp["div_level"]
SIG_MAP = sp["sigma"]
STRESS_COLS = [f"total_stress_{g}" for g in GCMS]

# ── Per-latitude band × per-GCM: proportion of species present in each class ──
edges = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 76)   # 75 bins (as in step 05)
ctr = (edges[:-1] + edges[1:]) / 2
band_rows = []
for i in range(len(edges) - 1):
    m = (df["Latitude"] >= edges[i]) & (df["Latitude"] < edges[i + 1])
    d = df[m]
    if d.empty:
        continue
    present = d["species"].unique()
    n = len(present)

    # per species: median stress over its grid points, for each GCM
    stress_sg = {s: {g: d.loc[d["species"] == s, f"total_stress_{g}"].median() for g in GCMS}
                 for s in present}

    # central (median-across-GCM) classification -> stacked proportions (sum to 1)
    cnt = dict.fromkeys(CLASSES, 0)
    for s in present:
        med_stress = float(np.median([stress_sg[s][g] for g in GCMS]))
        cnt[_classify(sp.loc[s, "div_level"], med_stress, sp.loc[s, "sigma"])] += 1
    row = {"lat_center": ctr[i], "n_species": n, **{c: cnt[c] / n for c in CLASSES}}

    # per-GCM vulnerable (Latent Vuln + Critical) and Critical fractions
    vuln_g, crit_g = [], []
    for g in GCMS:
        cv = cc = 0
        for s in present:
            cls = _classify(sp.loc[s, "div_level"], stress_sg[s][g], sp.loc[s, "sigma"])
            if cls in ("Latent Vulnerability", "Critical"):
                cv += 1
            if cls == "Critical":
                cc += 1
        vuln_g.append(cv / n); crit_g.append(cc / n)
    row["vuln_med"], row["vuln_lo"], row["vuln_hi"] = np.median(vuln_g), np.min(vuln_g), np.max(vuln_g)
    row["crit_med"] = np.median(crit_g)

    # POINT-weighted composition (same rule; unit = species×location points).
    # Each point classified at its own median-across-GCM stress.
    pt_stress = d[STRESS_COLS].median(axis=1).values
    pt_labels = classify_resilience(d["species"].map(DIV_MAP).values,
                                    pt_stress,
                                    d["species"].map(SIG_MAP).values)
    npt = len(d)
    row["n_points"] = npt
    for c in CLASSES:
        row[f"pt_{c}"] = (pt_labels == c).sum() / npt
    band_rows.append(row)
summary = pd.DataFrame(band_rows)
summary.to_csv(PROC / "species_vuln_location_summary.csv", index=False)

# ── Figure ──
fig = plt.figure(figsize=(18, 8.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.4, wspace=0.28)

# Panel A: median-GCM stacked proportion of species by latitude
axA = fig.add_subplot(gs[:, 0])
base = np.zeros(len(summary))
for c in CLASSES:
    axA.fill_between(summary["lat_center"], base, base + summary[c],
                     color=CLASS_COL[c], alpha=0.85, label=c)
    base = base + summary[c]

# Faint overlay: POINT-weighted composition (same class order), drawn as the
# cumulative class boundaries so divergence from the filled species-equal bands
# shows where point-weighting (area coverage) disagrees with counting species.
cum = np.zeros(len(summary))
for k, c in enumerate(CLASSES[:-1]):        # 3 internal boundaries; top = 1
    cum = cum + summary[f"pt_{c}"].values
    axA.plot(summary["lat_center"], cum, color="black", lw=1.1, ls="--", alpha=0.55,
             label="Point-weighted boundaries" if k == 0 else None, zorder=5)

axcount = axA.twinx()
axcount.plot(summary["lat_center"], summary["n_species"], "k-", lw=2, alpha=0.6)
axcount.set_ylabel("Number of species present", fontsize=12)
axcount.set_ylim(0, summary["n_species"].max() * 1.15)
axA.set_xlim(summary["lat_center"].min(), summary["lat_center"].max()); axA.set_ylim(0, 1)
axA.set_xlabel("Latitude", fontsize=12)
axA.set_ylabel("Proportion of species present (median GCM)", fontsize=12)
axA.set_title("Proportion of species in each vulnerability level by latitude\n"
              "(species-equal; diversity = species-median rank; stress = local Δ-climate + LULC, median GCM)",
              fontsize=13, fontweight="bold")
axA.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.95)
axA.grid(True, alpha=0.2, axis="x")

# Panel B: between-GCM range of the vulnerable fraction
axB = fig.add_subplot(gs[0, 1])
axB.plot(summary["lat_center"], summary["vuln_med"], "-", color="#c0392b", lw=2,
         label="Latent Vuln.+Critical (median)")
axB.fill_between(summary["lat_center"], summary["vuln_lo"], summary["vuln_hi"],
                 color="#c0392b", alpha=0.18, label="Between-GCM range")
axB.plot(summary["lat_center"], summary["crit_med"], "--", color="#7b241c", lw=1.6,
         label="Critical (median)")
axB.set_xlim(summary["lat_center"].min(), summary["lat_center"].max()); axB.set_ylim(0, 1)
axB.set_xlabel("Latitude", fontsize=11)
axB.set_ylabel("Proportion of species", fontsize=11)
axB.set_title("Vulnerable fraction with GCM range", fontsize=12, fontweight="bold")
axB.legend(fontsize=8, loc="upper left", framealpha=0.95)
axB.grid(True, alpha=0.2)

# Panel C: species median-diversity rank (the axis driving the split)
axC = fig.add_subplot(gs[1, 1])
spb = sp.sort_values("div_level")
ypos = np.arange(len(spb))
colors = ["#8e44ad" if mt == "pi" else "#2980b9" for mt in spb["metric"]]
axC.barh(ypos, spb["div_level"], color=colors, alpha=0.85)
axC.axvline(0.5, color="k", ls="--", lw=1.2, alpha=0.7)
axC.set_yticks(ypos)
axC.set_yticklabels([f"{s}" for s in spb.index], fontsize=6.5)
axC.set_xlim(0, 1)
axC.set_xlabel("Species median diversity (within-metric rank, 0–1)", fontsize=10)
axC.set_title("Species diversity rank (He = blue, π = purple)", fontsize=12, fontweight="bold")
axC.grid(True, alpha=0.2, axis="x")

for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig.savefig(FIG / f"species_vuln_location_summary.{ext}", bbox_inches="tight", **kw)
plt.close(fig)
print("\nSaved: figures/species_vuln_location_summary.{pdf,png}")
print("Saved: data/processed/species_vuln_location_summary.csv")

# ── Standalone POINT-WEIGHTED composition (unit = species×location cells) ──
fig2, axP = plt.subplots(figsize=(11, 7))
base = np.zeros(len(summary))
for c in CLASSES:
    axP.fill_between(summary["lat_center"], base, base + summary[f"pt_{c}"],
                     color=CLASS_COL[c], alpha=0.85, label=c)
    base = base + summary[f"pt_{c}"]
axpc = axP.twinx()
axpc.plot(summary["lat_center"], summary["n_points"], "k-", lw=2, alpha=0.6)
axpc.set_ylabel("Number of predictions (cells)", fontsize=12)
axpc.set_ylim(0, summary["n_points"].max() * 1.15)
axP.set_xlim(summary["lat_center"].min(), summary["lat_center"].max()); axP.set_ylim(0, 1)
axP.set_xlabel("Latitude", fontsize=12)
axP.set_ylabel("Proportion of predictions (cells)", fontsize=12)
axP.set_title("Point-weighted vulnerability composition by latitude\n"
              "(unit = species×location cells; diversity = species-median rank; median-GCM stress)",
              fontsize=13, fontweight="bold")
axP.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.95)
axP.grid(True, alpha=0.2, axis="x")
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig2.savefig(FIG / f"pointweighted_vuln_location_summary.{ext}", bbox_inches="tight", **kw)
plt.close(fig2)
print("Saved: figures/pointweighted_vuln_location_summary.{pdf,png}")

#!/usr/bin/env python3
"""Standalone reproduction of figures/ecosystem_resilience.pdf (the 3-panel
ecosystem strategy from step 05), rebuilt from the saved vulnerability_scores.csv
so it needs no model inference — with Panel A circles 5x larger.

Panel A: coast map, color = median vulnerability per latitude bin, marker size ∝
         species coverage (5x the original scaling).
Panel B: point-weighted resilience-class composition by latitude (Critical→
         Latent→At Risk→Resilient) + species-count line.
Panel C: vulnerability components by latitude (diversity deficit, climate
         exposure, model uncertainty, vulnerability score).

Output: figures/ecosystem_resilience_bigcircles.{pdf,png}
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from pathlib import Path

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
import ca_basemap as _cabm

df = pd.read_csv(PROC / "vulnerability_scores.csv")

# ── 75-bin latitudinal aggregation (as in step 05) ──
edges = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 76)
ctr = (edges[:-1] + edges[1:]) / 2
rows = []
for i in range(len(edges) - 1):
    pts = df[(df["Latitude"] >= edges[i]) & (df["Latitude"] < edges[i + 1])]
    if len(pts) == 0:
        continue
    n = len(pts)
    rows.append({
        "lat_center": ctr[i],
        "lon_center": pts["Longitude"].mean(),
        "n_species": pts["species"].nunique(),
        "mean_vulnerability": pts["vulnerability"].mean(),
        "median_vulnerability": pts["vulnerability"].median(),
        "mean_diversity": pts["diversity_norm"].mean(),
        "mean_exposure": pts["climate_exposure"].mean(),
        "mean_uncertainty": pts["sigma_norm"].mean(),
        "pct_critical": (pts["resilience_class"] == "Critical").sum() / n,
        "pct_latent": (pts["resilience_class"] == "Latent Vulnerability").sum() / n,
        "pct_at_risk": (pts["resilience_class"] == "At Risk").sum() / n,
        "pct_resilient": (pts["resilience_class"] == "Resilient").sum() / n,
    })
eco_df = pd.DataFrame(rows)

fig = plt.figure(figsize=(28, 24))
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.25, wspace=0.2)

# ─── Panel A: map ───
ax_map = fig.add_subplot(gs[:, 0])
CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
ca_xmin, ca_ymin = tr.transform(CA_LON_MIN, CA_LAT_MIN)
ca_xmax, ca_ymax = tr.transform(CA_LON_MAX, CA_LAT_MAX)
_cabm.add_land(ax_map, ca_xmin, ca_xmax, ca_ymin, ca_ymax)

vuln_cmap = LinearSegmentedColormap.from_list("vuln", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
eco_gdf = gpd.GeoDataFrame(
    eco_df, geometry=gpd.points_from_xy(eco_df["lon_center"], eco_df["lat_center"]),
    crs="EPSG:4326").to_crs(epsg=3857)

SCALE = 5.0   # 5x bigger circles
sizes = (eco_df["n_species"] / eco_df["n_species"].max() * 200 + 50) * SCALE
sc = ax_map.scatter(eco_gdf.geometry.x, eco_gdf.geometry.y,
                    c=eco_df["median_vulnerability"], cmap=vuln_cmap,
                    s=sizes, alpha=0.85, edgecolors="black", linewidth=0.4,
                    vmin=0.1, vmax=0.7, zorder=3)
cb = plt.colorbar(sc, ax=ax_map, shrink=0.5, pad=0.02)
cb.ax.tick_params(labelsize=11); cb.set_label("Median Vulnerability Score", fontsize=13)
ax_map.set_xlim(ca_xmin, ca_xmax); ax_map.set_ylim(ca_ymin, ca_ymax)
ax_map.set_title("Ecosystem Resilience\n(All Species Aggregated)", fontsize=16, fontweight="bold")
ax_map.set_xticks([]); ax_map.set_yticks([])
for ns, label in [(5, "5 spp"), (15, "15 spp"), (25, "25 spp")]:
    s = (ns / eco_df["n_species"].max() * 200 + 50) * SCALE
    ax_map.scatter([], [], s=s, c="gray", alpha=0.6, edgecolors="black", linewidth=0.4, label=label)
ax_map.legend(title="Species coverage", loc="lower left", fontsize=11,
              title_fontsize=12, frameon=True, fancybox=True)

# ─── Panel B: resilience-class composition ───
ax_stack = fig.add_subplot(gs[0, 1])
ax_stack.fill_between(eco_df["lat_center"], 0, eco_df["pct_critical"], color="#e74c3c", alpha=0.8, label="Critical")
ax_stack.fill_between(eco_df["lat_center"], eco_df["pct_critical"],
                      eco_df["pct_critical"] + eco_df["pct_latent"], color="#e67e22", alpha=0.8, label="Latent Vulnerability")
ax_stack.fill_between(eco_df["lat_center"], eco_df["pct_critical"] + eco_df["pct_latent"],
                      eco_df["pct_critical"] + eco_df["pct_latent"] + eco_df["pct_at_risk"], color="#f1c40f", alpha=0.8, label="At Risk")
ax_stack.fill_between(eco_df["lat_center"], eco_df["pct_critical"] + eco_df["pct_latent"] + eco_df["pct_at_risk"],
                      1.0, color="#2ecc71", alpha=0.8, label="Resilient")
axn = ax_stack.twinx()
axn.plot(eco_df["lat_center"], eco_df["n_species"], "k-", lw=2, alpha=0.6)
axn.set_ylabel("Number of species", fontsize=12)
ax_stack.set_xlabel("Latitude", fontsize=12); ax_stack.set_ylabel("Proportion of species", fontsize=12)
ax_stack.set_title("Resilience Class Composition Along Coast", fontsize=14, fontweight="bold")
ax_stack.set_ylim([0, 1]); ax_stack.legend(loc="upper right", fontsize=10, frameon=True)
ax_stack.grid(True, alpha=0.2, axis="x")

# ─── Panel C: vulnerability components ───
ax_comp = fig.add_subplot(gs[1, 1])
ax_comp.plot(eco_df["lat_center"], 1 - eco_df["mean_diversity"], "b-o", ms=3, lw=2, alpha=0.7, label="Diversity deficit (1 - div_norm)")
ax_comp.plot(eco_df["lat_center"], eco_df["mean_exposure"], "r-s", ms=3, lw=2, alpha=0.7, label="Climate exposure")
ax_comp.plot(eco_df["lat_center"], eco_df["mean_uncertainty"], "gray", ls="--", lw=1.5, alpha=0.6, label="Model uncertainty")
ax_comp.plot(eco_df["lat_center"], eco_df["mean_vulnerability"], "k-", lw=2.5, alpha=0.9, label="Vulnerability score")
ax_comp.set_xlabel("Latitude", fontsize=12); ax_comp.set_ylabel("Score (0-1)", fontsize=12)
ax_comp.set_title("Vulnerability Components Along Coast", fontsize=14, fontweight="bold")
ax_comp.legend(fontsize=10, loc="best", frameon=True); ax_comp.grid(True, alpha=0.3); ax_comp.set_ylim([0, 1])

fig.suptitle(f"Ecosystem-Level Climate Change Resilience — California Coast\n"
             f"{df['species'].nunique()} species, {len(df)} predictions",
             fontsize=20, fontweight="bold", y=0.98)
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig.savefig(FIG / f"ecosystem_resilience_bigcircles.{ext}", bbox_inches="tight", **kw)
plt.close()
print("Saved: figures/ecosystem_resilience_bigcircles.{pdf,png}")

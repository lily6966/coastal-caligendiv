#!/usr/bin/env python3
"""Combined 2-panel present-day resilience figure:

  Panel A — per-GCM model-agreement map (from resilience_agreement_perGCM):
      each coastal population colored by its modal resilience class across the
      5 GCMs, opacity = fraction of GCMs backing that class, black rings where
      no class holds a 3/5 majority.
  Panel B — resilience-class composition along the coast (from ecosystem_
      resilience): point-weighted proportion of predictions in each class by
      latitude (75 bins), with the number of species as a black line.

Both are present-day, same grid and classification (within-species diversity +
Δ-climate+LULC stress); Panel A resolves the per-GCM spread, Panel B the
along-coast composition.

Output: figures/resilience_map_composition.{pdf,png}
"""
import os
import sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from pathlib import Path

ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
sys.path.insert(0, str(ROOT))
import ca_basemap as _cabm

GCMS_N = 5
CLASSES = ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]
CLASS_COL = {"Resilient": "#2ecc71", "At Risk": "#f1c40f",
             "Latent Vulnerability": "#e67e22", "Critical": "#e74c3c"}

agree = pd.read_csv(PROC / "resilience_agreement_perGCM.csv")   # Panel A source
vs = pd.read_csv(PROC / "vulnerability_scores.csv")             # Panel B source

CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
ca_xmin, ca_ymin = tr.transform(CA_LON_MIN, CA_LAT_MIN)
ca_xmax, ca_ymax = tr.transform(CA_LON_MAX, CA_LAT_MAX)

fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.16)

# ─── Panel A: model-agreement map ───
axm = fig.add_subplot(gs[0, 0])
_cabm.add_land(axm, ca_xmin, ca_xmax, ca_ymin, ca_ymax)
pts = gpd.GeoDataFrame(
    agree, geometry=gpd.points_from_xy(agree["Longitude"], agree["Latitude"]),
    crs="EPSG:4326").to_crs(epsg=3857)
alpha = np.interp(agree["model_agreement"], [0.4, 1.0], [0.25, 0.95])
face = [to_rgba(CLASS_COL[c], a) for c, a in zip(agree["modal_class_perGCM"], alpha)]
axm.scatter(pts.geometry.x, pts.geometry.y, c=face, s=120, edgecolors="none", zorder=3)
contested = (agree["model_agreement"] < 0.6).values
axm.scatter(pts.geometry.x[contested], pts.geometry.y[contested],
            facecolors="none", edgecolors="black", s=240, linewidth=1.1, zorder=4)
axm.set_xlim(ca_xmin, ca_xmax); axm.set_ylim(ca_ymin, ca_ymax)
axm.set_xticks([]); axm.set_yticks([])
axm.set_title("Per-GCM resilience classification — model-agreement map\n"
              "(color = modal class across 5 GCMs; opacity = agreement)",
              fontsize=15, fontweight="bold")
leg = [Patch(facecolor=CLASS_COL[c], label=c) for c in CLASSES]
leg.append(Line2D([], [], marker="o", color="black", mfc="none", ls="none",
                  ms=12, label="No majority (<3/5)"))
axm.legend(handles=leg, loc="lower left", fontsize=11, frameon=True, framealpha=0.95)

# ─── Panel B: resilience-class composition along coast ───
axs = fig.add_subplot(gs[0, 1])
edges = np.linspace(vs["Latitude"].min(), vs["Latitude"].max(), 76)   # 75 bins
ctr = (edges[:-1] + edges[1:]) / 2
rows = []
for i in range(len(edges) - 1):
    b = vs[(vs["Latitude"] >= edges[i]) & (vs["Latitude"] < edges[i + 1])]
    if len(b) == 0:
        continue
    n = len(b)
    rows.append({"lat": ctr[i], "n_species": b["species"].nunique(),
                 "pct_critical": (b["resilience_class"] == "Critical").sum() / n,
                 "pct_latent": (b["resilience_class"] == "Latent Vulnerability").sum() / n,
                 "pct_at_risk": (b["resilience_class"] == "At Risk").sum() / n,
                 "pct_resilient": (b["resilience_class"] == "Resilient").sum() / n})
e = pd.DataFrame(rows)
axs.fill_between(e["lat"], 0, e["pct_critical"], color=CLASS_COL["Critical"], alpha=0.8, label="Critical")
axs.fill_between(e["lat"], e["pct_critical"], e["pct_critical"] + e["pct_latent"],
                 color=CLASS_COL["Latent Vulnerability"], alpha=0.8, label="Latent Vulnerability")
axs.fill_between(e["lat"], e["pct_critical"] + e["pct_latent"],
                 e["pct_critical"] + e["pct_latent"] + e["pct_at_risk"],
                 color=CLASS_COL["At Risk"], alpha=0.8, label="At Risk")
axs.fill_between(e["lat"], e["pct_critical"] + e["pct_latent"] + e["pct_at_risk"], 1.0,
                 color=CLASS_COL["Resilient"], alpha=0.8, label="Resilient")
axn = axs.twinx()
axn.plot(e["lat"], e["n_species"], "k-", lw=2, alpha=0.6)
axn.set_ylabel("Number of species", fontsize=12)
axs.set_xlim(e["lat"].min(), e["lat"].max()); axs.set_ylim(0, 1)
axs.set_xlabel("Latitude", fontsize=12); axs.set_ylabel("Proportion of species", fontsize=12)
axs.set_title("Resilience Class Composition Along Coast\n(present-day, point-weighted)",
              fontsize=15, fontweight="bold")
axs.legend(loc="upper right", fontsize=10, frameon=True)
axs.grid(True, alpha=0.2, axis="x")

fig.suptitle("Present-day genetic resilience — California Coast\n"
             f"{vs['species'].nunique()} species, {len(vs)} predictions | "
             "map: per-GCM classification & agreement | composition: along-coast class mix",
             fontsize=17, fontweight="bold", y=1.01)
for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
    fig.savefig(FIG / f"resilience_map_composition.{ext}", bbox_inches="tight", **kw)
plt.close()
print("Saved: figures/resilience_map_composition.{pdf,png}")

#!/usr/bin/env python3
"""
Improved per-species diversity map: Historical | Future | Change (ΔHe/Δπ), driven
by the per-GCM 1 km ensemble (gcm_ensemble_diversity.csv). Larger markers, a light
Natural-Earth coastline for context, and a diverging colormap for the change panel.

Outputs: figures/california_diversity_change_map_<species>.{pdf,png}
"""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")

CA_BBOX = (-125.5, 29.5, -114.5, 43.0)   # lon_min, lat_min, lon_max, lat_max
SPECIES = ["Eucyclogobius newberryi", "Zostera marina", "Macrocystis pyrifera",
           "Emerita analoga"]

_land = None
def land():
    global _land
    if _land is None:
        _land = gpd.clip(gpd.read_file(ROOT / "data" / "naturalearth" / "ne_10m_land.shp"),
                         box(*CA_BBOX))
    return _land

def _basemap(ax):
    land().plot(ax=ax, facecolor="#ededed", edgecolor="#9a9a9a", linewidth=0.5, zorder=0)
    ax.set_facecolor("#f4fafe")
    ax.set_xlim(CA_BBOX[0], CA_BBOX[2]); ax.set_ylim(CA_BBOX[1], CA_BBOX[3])
    ax.set_xlabel("Longitude"); ax.grid(True, alpha=0.25)


def main():
    g = pd.read_csv(PROC / "gcm_ensemble_diversity.csv")
    obs_all = _m.filter_ca_species(pd.read_csv(PROC / _pcali()), verbose=False)

    for sp in SPECIES:
        d = g[g["species"] == sp]
        if len(d) == 0:
            print(f"  {sp}: not in ensemble csv, skipping"); continue
        mt = d["metric_type"].iloc[0]
        lab = "π" if mt == "pi" else "He"
        obs = obs_all[obs_all["species"] == sp]
        pres, fut = d["present_mu"].values, d["future_mu_mean"].values
        chg = fut - pres

        vmin = min(pres.min(), fut.min()); vmax = max(pres.max(), fut.max())
        cmax = np.nanpercentile(np.abs(chg), 98) or 1e-6

        fig, axes = plt.subplots(1, 3, figsize=(16.5, 8.5))
        # Historical
        _basemap(axes[0])
        sc = axes[0].scatter(d["Longitude"], d["Latitude"], c=pres, cmap="viridis",
                             s=70, vmin=vmin, vmax=vmax, edgecolors="none", zorder=2)
        axes[0].scatter(obs["Longitude"], obs["Latitude"], c=obs["gen_div"], cmap="viridis",
                        s=95, vmin=vmin, vmax=vmax, edgecolors="red", linewidth=1.3, zorder=3)
        axes[0].set_ylabel("Latitude"); axes[0].set_title(f"Historical {lab}")
        plt.colorbar(sc, ax=axes[0], shrink=0.6, label=lab)
        # Future
        _basemap(axes[1])
        sc = axes[1].scatter(d["Longitude"], d["Latitude"], c=fut, cmap="viridis",
                             s=70, vmin=vmin, vmax=vmax, edgecolors="none", zorder=2)
        axes[1].set_title(f"Future {lab} (SSP5-8.5, 2081–2100, 5-GCM mean)")
        plt.colorbar(sc, ax=axes[1], shrink=0.6, label=lab)
        # Change
        _basemap(axes[2])
        sc = axes[2].scatter(d["Longitude"], d["Latitude"], c=chg, cmap="RdBu",
                             s=70, vmin=-cmax, vmax=cmax, edgecolors="none", zorder=2)
        axes[2].set_title(f"Change (future − historical {lab})")
        plt.colorbar(sc, ax=axes[2], shrink=0.6, label=f"Δ{lab}")

        fig.suptitle(f"Predicted genetic diversity, historical vs projected 2100 — {sp}\n"
                     f"mean {lab}: {pres.mean():.3f} → {fut.mean():.3f} "
                     f"(Δ {fut.mean()-pres.mean():+.3f}); red-edged = observed",
                     fontsize=13, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        stub = sp.replace(" ", "_")
        for ext, kw in (("pdf", {}), ("png", {"dpi": 170})):
            fig.savefig(FIG / f"california_diversity_change_map_{stub}.{ext}",
                        bbox_inches="tight", **kw)
        plt.close(fig)
        print(f"Saved: figures/california_diversity_change_map_{stub}.{{pdf,png}}  "
              f"(Δ mean {fut.mean()-pres.mean():+.3f})")


def _pcali():
    _p = __import__("04_predict_california")
    return _p.CALI_FILE


if __name__ == "__main__":
    main()

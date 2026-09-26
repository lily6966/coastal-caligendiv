#!/usr/bin/env python3
"""
Step 8: Diagnostics for the exposure definition — what the projected CHANGE in
each climate parameter looks like along the California coast.

Exposure itself is defined once, in climate_delta.py, and is used by step 5
onward; this script only opens it up so the composite can be read component by
component. It adds one quantity the composite does not carry: along-shore
climate velocity, the speed at which a present-day isotherm must travel up the
coast to stay in the same conditions.

Outputs:
  data/processed/climate_delta_grid.csv     per-component change on the coastal grid
  figures/delta_climate_profiles.{png,pdf}  latitudinal profiles of the change signal
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG_DIR = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
sys.path.insert(0, str(ROOT))
import climate_delta
_p = __import__("04_predict_california")

YEARS = 65.0                # marine baseline 2020 -> scenario midpoint 2085
GRAD_FLOOR = 0.002          # degC/km, keeps velocity finite on a flat stretch


def alongshore_distance_km(lats, lons):
    R = 6371.0
    la, lo = np.radians(lats), np.radians(lons)
    a = (np.sin(np.diff(la) / 2) ** 2 +
         np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2)
    return np.concatenate([[0.0], np.cumsum(2 * R * np.arcsin(np.sqrt(a)))])


def climate_velocity(baseline, delta, dist_km, years=YEARS, floor=GRAD_FLOOR):
    baseline = pd.Series(baseline).interpolate(limit_direction="both").values
    grad = np.maximum(np.abs(np.gradient(baseline, dist_km)), floor)
    grad = np.convolve(np.pad(grad, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
    return np.abs(delta) / years / grad


def main():
    lats, lons = _p.make_coastal_grid()
    grid = pd.DataFrame({"Latitude": lats, "Longitude": lons})
    dist_km = alongshore_distance_km(lats, lons)

    print("Evaluating the exposure definition on the coastal grid...")
    exposure, comp, delta = climate_delta.compute_climate_exposure(
        grid, return_components=True)
    mean, sd, lo, hi, per_gcm = climate_delta.exposure_ensemble(grid)

    pres, _ = climate_delta.marine_change(lats, lons)
    vel = climate_velocity(pres["sst_mean"], delta["d_sst_mean"].values, dist_km)

    out = grid.copy()
    out["alongshore_km"] = dist_km
    for c in delta.columns:
        out[f"delta_{c}"] = delta[c].values
    for c in comp.columns:
        out[f"score_{c}"] = comp[c].values
    out["baseline_sst_mean"] = pres["sst_mean"]
    out["baseline_sst_range"] = pres["sst_range"]
    out["sst_velocity_km_per_yr"] = vel
    out["climate_exposure"] = mean
    out["climate_exposure_sd"] = sd
    out["climate_exposure_min"] = lo
    out["climate_exposure_max"] = hi
    for g, v in per_gcm.items():
        out[f"climate_exposure_{g}"] = v
    out.to_csv(PROC / "climate_delta_grid.csv", index=False)
    print(f"  saved: {PROC / 'climate_delta_grid.csv'}")

    print("\n  component            mean    rho(latitude)")
    from scipy.stats import spearmanr
    for c in comp.columns:
        print(f"    {c:16s} {comp[c].mean():6.3f}    "
              f"{spearmanr(comp[c], lats).statistic:+.2f}")
    print(f"\n  exposure {mean.mean():.3f} (south {mean[:20].mean():.3f} → "
          f"north {mean[-20:].mean():.3f}), inter-model sd {sd.mean():.3f}")
    print(f"  along-shore SST velocity {vel.mean():.1f} km/yr "
          f"[{vel.min():.1f}, {vel.max():.1f}]")

    # ── Profiles ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Projected climate CHANGE along the California coast "
                 "(historical baseline → SSP5-8.5 2081-2100)", fontsize=13)
    panels = [
        ("delta_d_sst_mean", "Δ mean SST (°C)", "firebrick"),
        ("delta_d_ph", "pH decline (units)", "purple"),
        ("delta_d_o2", "O$_2$ decline (mmol m$^{-3}$)", "teal"),
        ("delta_d_bio5", "Δ hottest-month air T (°C)", "darkorange"),
        ("sst_velocity_km_per_yr", "Along-shore SST velocity (km yr$^{-1}$)", "steelblue"),
    ]
    for ax, (col, label, color) in zip(axes.flat, panels):
        ax.plot(lats, out[col], color=color, lw=2)
        ax.set_xlabel("Latitude (°N)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    ax = axes.flat[5]
    ax.plot(lats, mean, color="black", lw=2, label="Ensemble mean")
    ax.fill_between(lats, lo, hi, color="gray", alpha=0.3, label="GCM range")
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Climate exposure (Δ climate)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext, kw in [("png", {"dpi": 200}), ("pdf", {})]:
        fig.savefig(FIG_DIR / f"delta_climate_profiles.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  saved: figures/delta_climate_profiles.png + .pdf")


if __name__ == "__main__":
    main()

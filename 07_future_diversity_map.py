#!/usr/bin/env python3
"""
Plot predicted genetic diversity map: current vs future climate.
Generates california_diversity_map_future.png
"""

import warnings, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
FeatureProcessor = _m.FeatureProcessor
ConditionalNeuralProcess = _m.ConditionalNeuralProcess

_p = __import__("04_predict_california")
load_model_and_processor = _p.load_model_and_processor
prepare_context = _p.prepare_context
predict_at_locations = _p.predict_at_locations
make_target_df = _p.make_target_df
make_coastal_grid = _p.make_coastal_grid

_f = __import__("06_future_climate")
extract_worldclim_gcm = _f.extract_worldclim_gcm
extract_future_marine_avg = _f.extract_future_marine_avg
interpolate_missing = _f.interpolate_missing
WORLDCLIM_BIOS = _f.WORLDCLIM_BIOS
GCMS = _f.GCMS


def main():
    model, processor = load_model_and_processor()
    cali_df = pd.read_csv(PROC / "california_env.csv")

    SP = "Eucyclogobius newberryi"
    MT = "He"
    obs = cali_df[cali_df["species"] == SP]

    coast_lats, coast_lons = make_coastal_grid(n_points=150)

    # ── Current prediction ──
    context = prepare_context(obs, processor)
    target_current = make_target_df(coast_lats, coast_lons, SP, MT, cali_df)
    mu_current, sigma_current = predict_at_locations(model, context, target_current, processor)

    # ── Future prediction: swap env covariates (ensemble mean of 5 GCMs) ──
    print("Building future coastal environment (ensemble)...")
    bio_cols = [f"bio{b}" for b in WORLDCLIM_BIOS]

    gcm_grids = []
    for gcm in GCMS:
        g = pd.DataFrame({"Latitude": coast_lats, "Longitude": coast_lons})
        g = extract_worldclim_gcm(g, gcm)
        g = interpolate_missing(g, bio_cols)
        gcm_grids.append(g)

    future_grid = pd.DataFrame({"Latitude": coast_lats, "Longitude": coast_lons})
    for col in bio_cols:
        future_grid[col] = np.mean([g[col].values for g in gcm_grids], axis=0)

    future_grid = extract_future_marine_avg(future_grid)

    # Derived features
    future_grid["abs_latitude"] = future_grid["Latitude"].abs()
    future_grid["elevation_m"] = np.nan
    future_grid["temp_seasonality_norm"] = future_grid["bio4"] / (future_grid["bio7"] * 100 + 1)
    future_grid["precip_extremity"] = future_grid["bio13"] / (future_grid["bio14"] + 1)
    for col in ["bio5", "bio14", "bio4"]:
        col_z = f"_{col}_z"
        mean = future_grid[col].mean()
        std = future_grid[col].std()
        future_grid[col_z] = (future_grid[col] - mean) / std if std > 0 else 0
    future_grid["climate_stress"] = (
        future_grid["_bio5_z"] - future_grid["_bio14_z"] + future_grid["_bio4_z"]
    ) / 3
    future_grid.drop(columns=["_bio5_z", "_bio14_z", "_bio4_z"], inplace=True)

    target_future = make_target_df(coast_lats, coast_lons, SP, MT, cali_df)
    env_swap_cols = [c for c in future_grid.columns if c not in ["Latitude", "Longitude"]]
    for col in env_swap_cols:
        if col in target_future.columns:
            target_future[col] = future_grid[col].values

    mu_future, sigma_future = predict_at_locations(model, context, target_future, processor)

    print(f"\nCurrent He: mean={mu_current.mean():.4f}, "
          f"range=[{mu_current.min():.4f}, {mu_current.max():.4f}]")
    print(f"Future He:  mean={mu_future.mean():.4f}, "
          f"range=[{mu_future.min():.4f}, {mu_future.max():.4f}]")
    print(f"Change:     mean={mu_future.mean() - mu_current.mean():+.4f}")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 20))

    all_vals = np.concatenate([mu_current, mu_future, obs["gen_div"].values])
    vmin = all_vals.min() - (all_vals.max() - all_vals.min()) * 0.05
    vmax = all_vals.max() + (all_vals.max() - all_vals.min()) * 0.05

    # Panel 1: Current
    sc1 = ax1.scatter(coast_lons, coast_lats, c=mu_current, cmap="viridis",
                      s=50, alpha=0.8, edgecolors="gray", linewidth=0.3,
                      vmin=vmin, vmax=vmax, zorder=2)
    ax1.scatter(obs["Longitude"], obs["Latitude"], c=obs["gen_div"], cmap="viridis",
                s=100, marker="o", edgecolors="red", linewidth=1.5,
                vmin=vmin, vmax=vmax, zorder=3)
    ax1.set_xlabel("Longitude", fontsize=12)
    ax1.set_ylabel("Latitude", fontsize=12)
    ax1.set_title(f"Current Climate (Historical)\n{SP}",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    plt.colorbar(sc1, ax=ax1, shrink=0.4, label="He")

    # Panel 2: Future
    sc2 = ax2.scatter(coast_lons, coast_lats, c=mu_future, cmap="viridis",
                      s=50, alpha=0.8, edgecolors="gray", linewidth=0.3,
                      vmin=vmin, vmax=vmax, zorder=2)
    ax2.scatter(obs["Longitude"], obs["Latitude"], c=obs["gen_div"], cmap="viridis",
                s=100, marker="o", edgecolors="red", linewidth=1.5,
                vmin=vmin, vmax=vmax, zorder=3)
    ax2.set_xlabel("Longitude", fontsize=12)
    ax2.set_ylabel("Latitude", fontsize=12)
    ax2.set_title(f"Future Climate (SSP5-8.5, 2081-2100, 5-GCM)\n{SP}",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.2)
    plt.colorbar(sc2, ax=ax2, shrink=0.4, label="He")

    plt.suptitle(
        f"Predicted Genetic Diversity: {SP}\n"
        f"(Red-edged = observed, circles = predicted)\n"
        f"Current mean He = {mu_current.mean():.3f}  |  "
        f"Future mean He = {mu_future.mean():.3f}  |  "
        f"Change = {mu_future.mean() - mu_current.mean():+.3f}",
        fontsize=13, fontweight="bold", y=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for _ext, _kw in (("png", {"dpi": 200}), ("pdf", {})):
        plt.savefig(FIG_DIR / f"california_diversity_map_future.{_ext}",
                    bbox_inches="tight", **_kw)
    plt.close()
    print(f"\nSaved: {FIG_DIR / 'california_diversity_map_future.png'}")


if __name__ == "__main__":
    main()

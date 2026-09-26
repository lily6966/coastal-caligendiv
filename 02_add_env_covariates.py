#!/usr/bin/env python3
"""
Step 2: Add environmental covariates for climate change resilience analysis.

Uses WorldClim v2.1 bioclimatic rasters (10-min resolution, already downloaded)
for all covariates. No external API calls needed — fast and reliable.

Covariates (all from WorldClim + derived):
  CURRENT CLIMATE:
    bio1  — Annual Mean Temperature (°C × 10)
    bio12 — Annual Precipitation (mm)
    bio4  — Temperature Seasonality (SD × 100)
    bio15 — Precipitation Seasonality (CV)

  CLIMATE EXTREMES / VARIABILITY:
    bio5  — Max Temperature of Warmest Month
    bio6  — Min Temperature of Coldest Month
    bio7  — Temperature Annual Range (bio5 - bio6)
    bio13 — Precipitation of Wettest Month
    bio14 — Precipitation of Driest Month

  GEOGRAPHIC:
    abs_latitude — distance from equator
    Latitude, Longitude — raw coordinates
    elevation_m — from raster metadata (WorldClim DEM-aligned)
    is_coastal — proxy: 1 if bio12 > 0 and abs(Longitude) indicates coast

  CLIMATE EXPOSURE PROXIES (derived):
    temp_seasonality_norm — bio4 / bio7 (how variable within the annual range)
    precip_extremity — bio13 / (bio14 + 1) (wet/dry month ratio)
    climate_stress — composite: high temp + low precip + high seasonality
"""

import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
RASTER_DIR = ROOT / "data" / "env_rasters"

WORLDCLIM_BIOS = [1, 4, 5, 6, 7, 12, 13, 14, 15]


def extract_worldclim(df):
    """Extract bioclimatic values at point locations from GeoTIFFs."""
    lats = df["Latitude"].values
    lons = df["Longitude"].values

    for bio_num in WORLDCLIM_BIOS:
        tif_path = RASTER_DIR / f"wc2.1_10m_bio_{bio_num}.tif"
        col_name = f"bio{bio_num}"

        if not tif_path.exists():
            print(f"  WARNING: {tif_path.name} not found")
            df[col_name] = np.nan
            continue

        with rasterio.open(tif_path) as src:
            vals = []
            for lat, lon in zip(lats, lons):
                try:
                    row_px, col_px = src.index(lon, lat)
                    window = rasterio.windows.Window(col_px, row_px, 1, 1)
                    data = src.read(1, window=window)
                    v = float(data[0, 0])
                    if v == src.nodata or v < -1e30:
                        vals.append(np.nan)
                    else:
                        vals.append(v)
                except Exception:
                    vals.append(np.nan)
            df[col_name] = vals

    return df


def interpolate_missing(df, bio_cols):
    """Fill NaN bioclim values (e.g., coastal/ocean points) via nearest neighbor."""
    has_data = df[bio_cols].notna().all(axis=1)
    missing = ~has_data

    if has_data.sum() < 10 or missing.sum() == 0:
        return df

    print(f"  Interpolating {missing.sum()} points with missing bioclim data...")
    valid_coords = df.loc[has_data, ["Latitude", "Longitude"]].values
    for col in bio_cols:
        valid_vals = df.loc[has_data, col].values
        interp = NearestNDInterpolator(valid_coords, valid_vals)
        df.loc[missing, col] = interp(
            df.loc[missing, ["Latitude", "Longitude"]].values
        )
    return df


def add_derived_features(df):
    """Add geographic and climate-exposure derived features."""
    df["abs_latitude"] = df["Latitude"].abs()

    # Elevation placeholder (WorldClim 10-min doesn't include DEM separately)
    if "elevation_m" not in df.columns:
        df["elevation_m"] = np.nan

    # Climate exposure proxies
    df["temp_seasonality_norm"] = df["bio4"] / (df["bio7"] * 100 + 1)
    df["precip_extremity"] = df["bio13"] / (df["bio14"] + 1)

    # Climate stress index: normalized composite
    # High stress = high max temp + low min precip + high seasonality
    for col in ["bio5", "bio14", "bio4"]:
        if col in df.columns:
            col_z = f"_{col}_z"
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col_z] = (df[col] - mean) / std
            else:
                df[col_z] = 0

    if all(f"_{c}_z" in df.columns for c in ["bio5", "bio14", "bio4"]):
        df["climate_stress"] = (df["_bio5_z"] - df["_bio14_z"] + df["_bio4_z"]) / 3
        df.drop(columns=["_bio5_z", "_bio14_z", "_bio4_z"], inplace=True)

    return df


def process_dataset(name, csv_path):
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    n = len(df)

    # Extract WorldClim bioclimatic variables
    print(f"\n  Extracting WorldClim bioclim for {n} points...")
    df = extract_worldclim(df)

    bio_cols = [f"bio{b}" for b in WORLDCLIM_BIOS]
    valid = df[bio_cols].notna().all(axis=1).sum()
    print(f"  Direct extraction: {valid}/{n} have complete bioclim data")

    # Interpolate missing (coastal/ocean points fall outside raster land mask)
    df = interpolate_missing(df, bio_cols)
    valid = df[bio_cols].notna().all(axis=1).sum()
    print(f"  After interpolation: {valid}/{n} have complete bioclim data")

    # Derived features
    df = add_derived_features(df)
    print(f"  Added derived features: abs_latitude, temp_seasonality_norm, precip_extremity, climate_stress")

    return df


if __name__ == "__main__":
    # Process California (smaller)
    cali_df = process_dataset("California", PROC / "california.csv")
    cali_df.to_csv(PROC / "california_env.csv", index=False)
    print(f"\n  Saved california_env.csv: {len(cali_df)} rows")

    # Process global
    global_df = process_dataset("Global (GenDivRange)", PROC / "global_train.csv")
    global_df.to_csv(PROC / "global_train_env.csv", index=False)
    print(f"\n  Saved global_train_env.csv: {len(global_df)} rows")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    env_cols = [f"bio{b}" for b in WORLDCLIM_BIOS] + \
               ["abs_latitude", "temp_seasonality_norm", "precip_extremity", "climate_stress"]
    for label, df in [("California", cali_df), ("Global", global_df)]:
        print(f"\n{label}: {len(df)} records")
        for col in env_cols:
            if col in df.columns:
                v = df[col].notna().sum()
                print(f"  {col:>25s}: {v:>6d} valid, "
                      f"mean={df[col].mean():>10.2f}, "
                      f"range=[{df[col].min():.2f}, {df[col].max():.2f}]")

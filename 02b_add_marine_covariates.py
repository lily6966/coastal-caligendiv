#!/usr/bin/env python3
"""
Step 2b: Add marine environmental covariates from Bio-ORACLE v2.2.

Downloads ocean surface variables via Bio-ORACLE ERDDAP for the geographic
extent of each dataset, then extracts values at population coordinates
using nearest-neighbor matching.

Marine covariates added:
  sst_mean      — Mean Sea Surface Temperature (°C)
  sst_max       — Max SST (°C)
  sst_range     — SST annual range (°C)
  salinity_mean — Mean salinity (PSU)
  chl_mean      — Mean chlorophyll-a (mg/m³)
  o2_mean       — Mean dissolved oxygen (mol/m³)
  ph_mean       — Mean ocean pH
  is_marine     — Binary indicator (1 if Bio-ORACLE data available)
"""

import json, warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"

BIO_ORACLE_VARS = {
    "thetao_baseline_2000_2019_depthsurf": [
        ("thetao_mean", "sst_mean"),
        ("thetao_max", "sst_max"),
        ("thetao_range", "sst_range"),
    ],
    "so_baseline_2000_2019_depthsurf": [
        ("so_mean", "salinity_mean"),
    ],
    "chl_baseline_2000_2018_depthsurf": [
        ("chl_mean", "chl_mean"),
    ],
    "o2_baseline_2000_2018_depthsurf": [
        ("o2_mean", "o2_mean"),
    ],
    "ph_baseline_2000_2018_depthsurf": [
        ("ph_mean", "ph_mean"),
    ],
}

CACHE_PATH = PROC / "bio_oracle_cache.json"


def fetch_bio_oracle_region(dataset, erddap_vars, lat_min, lat_max, lon_min, lon_max, cache):
    """Fetch Bio-ORACLE data for a lat/lon bounding box."""
    cache_key = f"{dataset}_{lat_min:.0f}_{lat_max:.0f}_{lon_min:.0f}_{lon_max:.0f}"
    if cache_key in cache:
        return pd.DataFrame(cache[cache_key])

    var_str = ",".join(
        f"{v}[(2010-01-01T00:00:00Z)][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
        for v, _ in erddap_vars
    )
    url = f"{ERDDAP_BASE}/{dataset}.json?{var_str}"

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        cols = data["table"]["columnNames"]
        rows = data["table"]["rows"]
        df = pd.DataFrame(rows, columns=cols)
        # Rename columns
        for erddap_name, our_name in erddap_vars:
            if erddap_name in df.columns:
                df.rename(columns={erddap_name: our_name}, inplace=True)
        # Keep only rows with at least one valid value
        value_cols = [our_name for _, our_name in erddap_vars]
        df = df[df[value_cols].notna().any(axis=1)]
        cache[cache_key] = df.to_dict(orient="list")
        return df
    except Exception as e:
        print(f"    Error fetching {dataset}: {e}")
        return pd.DataFrame()


def extract_marine_vars(pop_df, cache):
    """Extract Bio-ORACLE variables at population coordinates."""
    lats = pop_df["Latitude"].values
    lons = pop_df["Longitude"].values

    lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
    lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5

    # Fetch all Bio-ORACLE variables for the bounding box
    all_ocean = None
    for dataset, var_list in BIO_ORACLE_VARS.items():
        print(f"    Fetching {dataset}...")
        ocean_df = fetch_bio_oracle_region(
            dataset, var_list, lat_min, lat_max, lon_min, lon_max, cache
        )
        if len(ocean_df) == 0:
            continue

        if all_ocean is None:
            all_ocean = ocean_df[["latitude", "longitude"] +
                                 [v[1] for v in var_list]].copy()
        else:
            merge_cols = [v[1] for v in var_list]
            ocean_merge = ocean_df[["latitude", "longitude"] + merge_cols].copy()
            all_ocean = all_ocean.merge(
                ocean_merge, on=["latitude", "longitude"], how="outer"
            )

    if all_ocean is None or len(all_ocean) == 0:
        print("    No ocean data retrieved!")
        return pop_df

    # Build nearest-neighbor interpolators for each marine variable
    ocean_coords = all_ocean[["latitude", "longitude"]].values
    valid_mask = all_ocean.notna().all(axis=1) if len(all_ocean.columns) > 2 else np.ones(len(all_ocean), dtype=bool)

    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean"]

    for col in marine_cols:
        if col not in all_ocean.columns:
            pop_df[col] = np.nan
            continue

        valid = all_ocean[col].notna()
        if valid.sum() < 5:
            pop_df[col] = np.nan
            continue

        interp = NearestNDInterpolator(
            ocean_coords[valid.values],
            all_ocean.loc[valid, col].values
        )
        pop_df[col] = interp(np.column_stack([lats, lons]))

    # is_marine: 1 if nearest ocean grid point is within ~50km (~0.5°)
    from scipy.spatial import cKDTree
    tree = cKDTree(ocean_coords)
    dists, _ = tree.query(np.column_stack([lats, lons]))
    pop_df["is_marine"] = (dists < 0.5).astype(int)

    return pop_df


def process_dataset(name, csv_path, cache):
    print(f"\n{'='*60}")
    print(f"Adding marine covariates to {name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"  {n} records, lat range [{df['Latitude'].min():.1f}, {df['Latitude'].max():.1f}]")

    df = extract_marine_vars(df, cache)

    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean", "is_marine"]
    for col in marine_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            if col == "is_marine":
                marine_count = df[col].sum()
                print(f"  {col}: {marine_count}/{n} classified as marine")
            else:
                if valid > 0:
                    print(f"  {col}: {valid}/{n} valid, "
                          f"mean={df[col].mean():.3f}, "
                          f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")
                else:
                    print(f"  {col}: no valid data")

    return df


if __name__ == "__main__":
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached Bio-ORACLE regions")
    else:
        cache = {}

    # Process California first (all marine/coastal)
    cali_df = process_dataset("California", PROC / "california_env.csv", cache)
    cali_df.to_csv(PROC / "california_env.csv", index=False)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    # Process global (mixed — many terrestrial, some coastal)
    # For efficiency, chunk the global data by 10° latitude bands
    global_df = pd.read_csv(PROC / "global_train_env.csv")
    print(f"\n{'='*60}")
    print(f"Adding marine covariates to Global (GenDivRange)")
    print(f"{'='*60}")
    print(f"  {len(global_df)} records total")

    lat_bands = np.arange(-90, 91, 10)
    chunks = []
    for i in range(len(lat_bands) - 1):
        lat_lo, lat_hi = lat_bands[i], lat_bands[i + 1]
        mask = (global_df["Latitude"] >= lat_lo) & (global_df["Latitude"] < lat_hi)
        if mask.sum() == 0:
            continue
        chunk = global_df[mask].copy()
        print(f"\n  Band [{lat_lo}, {lat_hi}): {len(chunk)} records")
        chunk = extract_marine_vars(chunk, cache)
        chunks.append(chunk)

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

    global_out = pd.concat(chunks, ignore_index=True)

    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean", "is_marine"]
    print(f"\nGlobal marine covariate summary:")
    for col in marine_cols:
        if col in global_out.columns:
            valid = global_out[col].notna().sum()
            if col == "is_marine":
                print(f"  {col}: {global_out[col].sum()}/{len(global_out)} marine")
            elif valid > 0:
                print(f"  {col}: {valid}/{len(global_out)} valid")

    global_out.to_csv(PROC / "global_train_env.csv", index=False)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"\nCache saved: {len(cache)} regions")
    print("Done!")

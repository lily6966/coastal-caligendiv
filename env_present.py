#!/usr/bin/env python3
"""
Present-day environmental covariates sampled at arbitrary coordinates.

04_predict_california.make_target_df used to copy every covariate from the
NEAREST OBSERVED POPULATION, which caps the effective prediction resolution at
the spacing of the observation records however fine the transect is. This module
samples the same source layers the training data came from, directly at each
prediction point, so a finer transect carries real information:

  bio1..bio15   WorldClim v2.1, 30 arc-sec (~1 km) California crops where
                available, 10 arc-min (~18.5 km) global rasters elsewhere
  marine vars   Bio-ORACLE baseline 2000-2019 @0.05deg (~6 km), California box

Derived features follow 02_add_env_covariates.py exactly. climate_stress is a
z-score composite, so it is standardized against the CALIFORNIA TRAINING RECORDS
rather than against the prediction points — the model has to see it on the scale
it was trained on.

Species-level categorical features (Life_form, taxonomy, Marker_type, BIOME) are
not spatial and still come from the species' own records.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))

import climate_delta as _cd

BIOS = [1, 4, 5, 6, 7, 12, 13, 14, 15]

BASELINE_DATASETS = {
    "thetao_baseline_2000_2019_depthsurf": [
        ("thetao_mean", "sst_mean"), ("thetao_max", "sst_max"),
        ("thetao_range", "sst_range")],
    "so_baseline_2000_2019_depthsurf": [("so_mean", "salinity_mean")],
    "chl_baseline_2000_2018_depthsurf": [("chl_mean", "chl_mean")],
    "o2_baseline_2000_2018_depthsurf": [("o2_mean", "o2_mean")],
    "ph_baseline_2000_2018_depthsurf": [("ph_mean", "ph_mean")],
}
BASELINE_TIME = "2010-01-01T00:00:00Z"
MARINE_VARS = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
               "chl_mean", "o2_mean", "ph_mean"]

_marine_interp = {}


def _marine_interpolators():
    """Nearest-neighbour interpolators for the observational marine baseline."""
    if _marine_interp:
        return _marine_interp
    from scipy.interpolate import NearestNDInterpolator
    for dataset, var_list in BASELINE_DATASETS.items():
        df = _cd._fetch(dataset, var_list, BASELINE_TIME, _cd.CA_BOX, 1)
        coords = df[["latitude", "longitude"]].values
        for _, name in var_list:
            if name not in df.columns:
                continue
            ok = df[name].notna().values
            if ok.sum() >= 5:
                _marine_interp[name] = NearestNDInterpolator(
                    coords[ok], df.loc[ok, name].values)
    return _marine_interp


def present_env(lats, lons, cali_df=None):
    """WorldClim + Bio-ORACLE covariates at each (lat, lon), plus derived features."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    out = pd.DataFrame({"Latitude": lats, "Longitude": lons})

    # ── Terrestrial: sample the rasters directly, finest resolution available ──
    for bio in BIOS:
        out[f"bio{bio}"] = _cd.sample_bio(bio, lats, lons)

    # ── Marine: observational baseline, the same layers 02b extracted ──
    interps = _marine_interpolators()
    pts = np.column_stack([lats, lons])
    for var in MARINE_VARS:
        out[var] = interps[var](pts) if var in interps else np.nan
    out["is_marine"] = 1

    # ── Derived features, following 02_add_env_covariates.py ──
    out["abs_latitude"] = np.abs(lats)
    out["elevation_m"] = np.nan
    out["temp_seasonality_norm"] = out["bio4"] / (out["bio7"] * 100 + 1)
    out["precip_extremity"] = out["bio13"] / (out["bio14"] + 1)

    ref = cali_df if cali_df is not None else out
    z = {}
    for col in ["bio5", "bio14", "bio4"]:
        mean, std = ref[col].mean(), ref[col].std()
        z[col] = (out[col] - mean) / std if std > 0 else 0.0
    out["climate_stress"] = (z["bio5"] - z["bio14"] + z["bio4"]) / 3

    return out


ENV_COLS = ([f"bio{b}" for b in BIOS] + MARINE_VARS +
            ["is_marine", "abs_latitude", "elevation_m",
             "temp_seasonality_norm", "precip_extremity", "climate_stress"])


if __name__ == "__main__":
    lats = np.linspace(30.5, 42.0, 8)
    lons = np.interp(lats, [30.5, 32.5, 33.0, 34.0, 34.5, 35.0, 36.5, 37.5, 38.5, 40.0, 42.0],
                     [-116.1, -117.2, -117.3, -118.5, -120.5, -120.7, -122.0, -122.5,
                      -123.0, -124.2, -124.4])
    cali = pd.read_csv(PROC / "california_env.csv")
    print(present_env(lats, lons, cali)[
        ["Latitude", "bio1", "bio5", "sst_mean", "ph_mean", "climate_stress"]
    ].round(3).to_string(index=False))

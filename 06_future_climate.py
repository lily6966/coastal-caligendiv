#!/usr/bin/env python3
"""
Step 6: Future climate vulnerability — multi-model ensemble.

Downloads future climate projections from:
  - WorldClim v2.1 CMIP6: 5 GCMs × SSP5-8.5 2081-2100 (terrestrial bioclim)
  - Bio-ORACLE SSP5-8.5 average of 2080+2090 decadal steps (marine)

Predicts future genetic diversity by feeding future environmental conditions
into the trained CNP model.

Exposure is the projected CHANGE in climate parameters (climate_delta.py, via
step 5), evaluated per GCM so the ensemble carries an inter-model spread. It is
the same quantity as step 5's climate_exposure — what differs here is the
diversity term, which comes from the CNP run under end-of-century climate.

Outputs:
  data/processed/vulnerability_scores_future.csv
  figures/ecosystem_resilience_future.pdf
  figures/div_vs_exposure_future.pdf
"""

import os
import json, os, sys, warnings, zipfile, io
import numpy as np
import pandas as pd
import requests
import rasterio
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG_DIR = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
RASTER_DIR = ROOT / "data" / "env_rasters"

sys.path.insert(0, str(ROOT))
_r = __import__("05_resilience_assessment")
compute_climate_exposure = _r.compute_climate_exposure
classify_resilience = _r.classify_resilience
GLOBAL_REF = _r.GLOBAL_REF

_p = __import__("04_predict_california")
make_coastal_grid = _p.make_coastal_grid
load_model_and_processor = _p.load_model_and_processor
prepare_context = _p.prepare_context
predict_at_locations = _p.predict_at_locations
make_target_df = _p.make_target_df

_m = __import__("03_train_model")
filter_ca_species = _m.filter_ca_species
# models/feature_processor.pkl was pickled from 03_train_model.py running as
# __main__, so these names must exist here for pickle.load to resolve them.
FeatureProcessor = _m.FeatureProcessor
ConditionalNeuralProcess = _m.ConditionalNeuralProcess

WORLDCLIM_BIOS = [1, 4, 5, 6, 7, 12, 13, 14, 15]
PERIOD = "2081-2100"

GCMS = [
    "BCC-CSM2-MR",
    "CanESM5",
    "CNRM-CM6-1",
    "IPSL-CM6A-LR",
    "MIROC6",
]

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"
FUTURE_TIMES = ["2080-01-01T00:00:00Z", "2090-01-01T00:00:00Z"]

BIO_ORACLE_FUTURE = {
    "thetao_ssp585_2020_2100_depthsurf": [
        ("thetao_mean", "sst_mean"),
        ("thetao_max", "sst_max"),
        ("thetao_range", "sst_range"),
    ],
    "so_ssp585_2020_2100_depthsurf": [
        ("so_mean", "salinity_mean"),
    ],
    "chl_ssp585_2020_2100_depthsurf": [
        ("chl_mean", "chl_mean"),
    ],
    "o2_ssp585_2020_2100_depthsurf": [
        ("o2_mean", "o2_mean"),
    ],
    "ph_ssp585_2020_2100_depthsurf": [
        ("ph_mean", "ph_mean"),
    ],
}


# ═══════════════════════════════════════════════════════════
# WorldClim future rasters — per GCM
# ═══════════════════════════════════════════════════════════

def _gcm_dir(gcm=None):
    d = RASTER_DIR / "future_ssp585"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_worldclim_gcm(gcm):
    outdir = _gcm_dir(gcm)
    tif = outdir / f"wc2.1_10m_bioc_{gcm}_ssp585_{PERIOD}.tif"
    if tif.exists():
        print(f"  {gcm}: already downloaded")
        return

    url = (f"https://geodata.ucdavis.edu/climate/worldclim/2_1/fut/10m/"
           f"wc2.1_10m_bioc_{gcm}_ssp585_{PERIOD}.zip")
    print(f"  {gcm}: downloading from {url}")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    data = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=65536):
        data.write(chunk)
        downloaded += len(chunk)
        if total > 0:
            print(f"\r    {downloaded/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()

    data.seek(0)
    with zipfile.ZipFile(data) as zf:
        for member in zf.namelist():
            if member.endswith('.tif'):
                target = outdir / os.path.basename(member)
                with open(target, 'wb') as f:
                    f.write(zf.read(member))
    print(f"    Extracted to {outdir}")


def extract_worldclim_gcm(df, gcm):
    """Extract bioclim from a single GCM's multi-band TIF."""
    outdir = _gcm_dir(gcm)
    tif_path = outdir / f"wc2.1_10m_bioc_{gcm}_ssp585_{PERIOD}.tif"
    if not tif_path.exists():
        alt = list(outdir.glob("*.tif"))
        tif_path = alt[0] if alt else None
    if tif_path is None:
        for bio_num in WORLDCLIM_BIOS:
            df[f"bio{bio_num}"] = np.nan
        return df

    lats = df["Latitude"].values
    lons = df["Longitude"].values

    with rasterio.open(tif_path) as src:
        for bio_num in WORLDCLIM_BIOS:
            vals = []
            for lat, lon in zip(lats, lons):
                try:
                    row_px, col_px = src.index(lon, lat)
                    window = rasterio.windows.Window(col_px, row_px, 1, 1)
                    band_data = src.read(bio_num, window=window)
                    v = float(band_data[0, 0])
                    if v == src.nodata or v < -1e30:
                        vals.append(np.nan)
                    else:
                        vals.append(v)
                except Exception:
                    vals.append(np.nan)
            df[f"bio{bio_num}"] = vals
    return df


# ═══════════════════════════════════════════════════════════
# Bio-ORACLE future — average of 2080 + 2090
# ═══════════════════════════════════════════════════════════

def fetch_bio_oracle_time(dataset, erddap_vars, lat_min, lat_max, lon_min, lon_max, time_str):
    var_str = ",".join(
        f"{v}[({time_str})][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
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
        for erddap_name, our_name in erddap_vars:
            if erddap_name in df.columns:
                df.rename(columns={erddap_name: our_name}, inplace=True)
        value_cols = [our_name for _, our_name in erddap_vars]
        df = df[df[value_cols].notna().any(axis=1)]
        return df
    except Exception as e:
        print(f"      Error fetching {dataset} @ {time_str}: {e}")
        return pd.DataFrame()


def extract_future_marine_avg(pop_df):
    """Fetch Bio-ORACLE for 2080 and 2090, average them."""
    lats = pop_df["Latitude"].values
    lons = pop_df["Longitude"].values
    lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
    lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5

    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean"]

    time_results = {}
    for t_str in FUTURE_TIMES:
        print(f"    Fetching Bio-ORACLE @ {t_str}...")
        all_ocean = None
        for dataset, var_list in BIO_ORACLE_FUTURE.items():
            ocean_df = fetch_bio_oracle_time(
                dataset, var_list, lat_min, lat_max, lon_min, lon_max, t_str
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
            continue

        ocean_coords = all_ocean[["latitude", "longitude"]].values
        result_df = pd.DataFrame(index=pop_df.index)

        for col in marine_cols:
            if col not in all_ocean.columns:
                result_df[col] = np.nan
                continue
            valid = all_ocean[col].notna()
            if valid.sum() < 5:
                result_df[col] = np.nan
                continue
            interp = NearestNDInterpolator(
                ocean_coords[valid.values],
                all_ocean.loc[valid, col].values
            )
            result_df[col] = interp(np.column_stack([lats, lons]))

        time_results[t_str] = result_df

    # Average across time steps
    if not time_results:
        for col in marine_cols:
            pop_df[col] = np.nan
        pop_df["is_marine"] = 0
        return pop_df

    for col in marine_cols:
        vals = [time_results[t][col] for t in time_results if col in time_results[t].columns]
        if vals:
            pop_df[col] = pd.concat(vals, axis=1).mean(axis=1)
        else:
            pop_df[col] = np.nan

    from scipy.spatial import cKDTree
    # Use last ocean grid for is_marine
    last_ocean = list(time_results.values())[-1]
    pop_df["is_marine"] = 1  # coastal grid, always marine

    return pop_df


# ═══════════════════════════════════════════════════════════
# Build future env grid for one GCM
# ═══════════════════════════════════════════════════════════

def interpolate_missing(df, bio_cols):
    has_data = df[bio_cols].notna().all(axis=1)
    missing = ~has_data
    if has_data.sum() < 10 or missing.sum() == 0:
        return df
    valid_coords = df.loc[has_data, ["Latitude", "Longitude"]].values
    for col in bio_cols:
        valid_vals = df.loc[has_data, col].values
        interp = NearestNDInterpolator(valid_coords, valid_vals)
        df.loc[missing, col] = interp(
            df.loc[missing, ["Latitude", "Longitude"]].values
        )
    return df


def build_gcm_coastal_env(gcm, coast_lats, coast_lons, marine_df):
    """Build future env for one GCM (terrestrial varies, marine shared)."""
    df = pd.DataFrame({"Latitude": coast_lats, "Longitude": coast_lons})
    df = extract_worldclim_gcm(df, gcm)

    bio_cols = [f"bio{b}" for b in WORLDCLIM_BIOS]
    df = interpolate_missing(df, bio_cols)

    # Copy shared marine columns
    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean", "is_marine"]
    for col in marine_cols:
        if col in marine_df.columns:
            df[col] = marine_df[col].values

    return df


def build_future_env_ensemble(coast_lats, coast_lons, marine_grid):
    """Build ensemble-mean future env grid with all model features."""
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

    marine_cols = ["sst_mean", "sst_max", "sst_range", "salinity_mean",
                   "chl_mean", "o2_mean", "ph_mean", "is_marine"]
    for col in marine_cols:
        if col in marine_grid.columns:
            future_grid[col] = marine_grid[col].values

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

    return future_grid


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MULTI-MODEL ENSEMBLE FUTURE CLIMATE ASSESSMENT")
    print(f"GCMs: {', '.join(GCMS)}")
    print(f"Terrestrial: WorldClim SSP5-8.5 {PERIOD}")
    print(f"Marine: Bio-ORACLE SSP5-8.5 avg({', '.join(FUTURE_TIMES)})")
    print("=" * 60)

    # ── Step 1: Download all GCM rasters ──
    print("\n── Downloading WorldClim future rasters ──")
    for gcm in GCMS:
        download_worldclim_gcm(gcm)

    # ── Step 2: Load current predictions ──
    current_results = pd.read_csv(PROC / "vulnerability_scores.csv")
    print(f"\nLoaded current results: {len(current_results)} records, "
          f"{current_results['species'].nunique()} species")

    coast_lats, coast_lons = make_coastal_grid()

    # ── Step 3: Fetch future marine (shared across GCMs) ──
    print("\n── Fetching future marine data (Bio-ORACLE, avg 2080+2090) ──")
    marine_grid = pd.DataFrame({"Latitude": coast_lats, "Longitude": coast_lons})
    marine_grid = extract_future_marine_avg(marine_grid)

    marine_cols_report = ["sst_mean", "sst_max", "sst_range", "ph_mean", "o2_mean"]
    print("  Marine averages (2080+2090):")
    for col in marine_cols_report:
        if col in marine_grid.columns:
            print(f"    {col}: {marine_grid[col].mean():.4f}")

    # ── Step 4: Compute change-based exposure per GCM ──
    print("\n── Computing Δ-climate exposure per GCM ──")
    from scipy.spatial import cKDTree

    grid_coords = np.column_stack([coast_lats, coast_lons])
    tree = cKDTree(grid_coords)
    pred_coords = np.column_stack([
        current_results["Latitude"].values,
        current_results["Longitude"].values
    ])
    _, grid_indices = tree.query(pred_coords)

    gcm_exposures = {}

    for gcm in GCMS:
        print(f"\n  Processing {gcm}...")
        # Exposure depends on the projected CHANGE at each location, not on the
        # future state, so it is evaluated straight from the coordinates.
        pred_env = current_results[["Latitude", "Longitude", "species", "metric_type"]].copy()
        exposure = compute_climate_exposure(pred_env, gcm=gcm)
        gcm_exposures[gcm] = exposure.values

        print(f"    Mean Δ-exposure: {exposure.mean():.4f}, "
              f"range: [{exposure.min():.4f}, {exposure.max():.4f}]")

    # ── Step 5: Ensemble statistics ──
    print("\n── Ensemble statistics ──")
    exposure_matrix = np.column_stack([gcm_exposures[g] for g in GCMS])
    ensemble_mean = exposure_matrix.mean(axis=1)
    ensemble_std = exposure_matrix.std(axis=1)
    ensemble_min = exposure_matrix.min(axis=1)
    ensemble_max = exposure_matrix.max(axis=1)

    print(f"  Ensemble mean exposure: {ensemble_mean.mean():.4f}")
    print(f"  Inter-model SD (mean):  {ensemble_std.mean():.4f}")
    print(f"  Inter-model range:      {(ensemble_max - ensemble_min).mean():.4f}")

    for gcm in GCMS:
        print(f"    {gcm:20s}: mean={gcm_exposures[gcm].mean():.4f}")

    # ── Step 6: Predict future genetic diversity ──
    print("\n── Predicting genetic diversity under future climate ──")
    model, processor = load_model_and_processor()
    # Context must carry LULC columns so the LULC model sees them (not zero-filled)
    cali_df = pd.read_csv(PROC / _p.CALI_FILE)
    cali_df = filter_ca_species(cali_df)

    future_env_grid = build_future_env_ensemble(coast_lats, coast_lons, marine_grid)
    env_swap_cols = [c for c in future_env_grid.columns if c not in ["Latitude", "Longitude"]]

    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    HE_REF_LO = global_df["gen_div"].quantile(0.02)
    HE_REF_HI = global_df["gen_div"].quantile(0.98)
    pi_vals = cali_df[cali_df["metric_type"] == "pi"]["gen_div"]
    PI_REF_LO = pi_vals.quantile(0.02)
    PI_REF_HI = pi_vals.quantile(0.98)

    future_pred_mu = np.full(len(current_results), np.nan)
    future_pred_sigma = np.full(len(current_results), np.nan)

    for sp in current_results["species"].unique():
        sp_mask = current_results["species"] == sp
        sp_rows = current_results[sp_mask]
        mt = sp_rows["metric_type"].iloc[0]
        sp_lats = sp_rows["Latitude"].values
        sp_lons = sp_rows["Longitude"].values

        obs = cali_df[cali_df["species"] == sp]
        context = prepare_context(obs, processor)

        target_df = make_target_df(sp_lats, sp_lons, sp, mt, cali_df)
        _, sp_grid_idx = tree.query(np.column_stack([sp_lats, sp_lons]))
        for col in env_swap_cols:
            if col in target_df.columns:
                target_df[col] = future_env_grid[col].values[sp_grid_idx]

        mu, sigma = predict_at_locations(model, context, target_df, processor)
        idx = np.where(sp_mask)[0]
        future_pred_mu[idx] = mu
        future_pred_sigma[idx] = sigma

        print(f"  {sp}: current He/pi={sp_rows['pred_mu'].mean():.4f} → "
              f"future={mu.mean():.4f} ({mu.mean() - sp_rows['pred_mu'].mean():+.4f})")

    # Future diversity on the cross-species reference — the scale the score uses.
    future_div_norm = np.full(len(current_results), np.nan)
    he_mask = (current_results["metric_type"] == "He").values
    pi_mask = (current_results["metric_type"] == "pi").values
    future_div_norm[he_mask] = np.clip(
        (future_pred_mu[he_mask] - HE_REF_LO) / (HE_REF_HI - HE_REF_LO), 0, 1)
    future_div_norm[pi_mask] = np.clip(
        (future_pred_mu[pi_mask] - PI_REF_LO) / (PI_REF_HI - PI_REF_LO), 0, 1)

    future_sigma_norm = np.zeros(len(current_results))
    for sp in current_results["species"].unique():
        sp_mask = current_results["species"] == sp
        svals = future_pred_sigma[sp_mask]
        smn, smx = svals.min(), svals.max()
        if smx - smn > 1e-8:
            future_sigma_norm[sp_mask] = (svals - smn) / (smx - smn)

    print(f"\n  Future diversity (norm): mean={np.nanmean(future_div_norm):.4f} "
          f"(current={current_results['diversity_norm'].mean():.4f})")

    # ── Step 7: Build final results ──
    future_results = current_results.copy()
    future_results["pred_mu_future"] = future_pred_mu
    future_results["pred_sigma_future"] = future_pred_sigma
    future_results["diversity_norm_future"] = future_div_norm
    future_results["sigma_norm_future"] = future_sigma_norm
    future_results["climate_exposure_future"] = ensemble_mean
    future_results["climate_exposure_future_std"] = ensemble_std
    future_results["climate_exposure_future_min"] = ensemble_min
    future_results["climate_exposure_future_max"] = ensemble_max
    for gcm in GCMS:
        future_results[f"exposure_{gcm}"] = gcm_exposures[gcm]

    # Vulnerability with future diversity + future exposure
    future_results["vulnerability_future"] = (
        (1 - future_results["diversity_norm_future"]) * future_results["climate_exposure_future"]
        + 0.3 * future_results["sigma_norm_future"]
    )
    future_results["resilience_class_future"] = classify_resilience(
        future_results["diversity_norm_future"].values,
        future_results["climate_exposure_future"].values,
        future_results["sigma_norm_future"].values,
    )

    # Uncertainty band using min/max exposure (diversity held at ensemble mean)
    future_results["vulnerability_future_lo"] = (
        (1 - future_results["diversity_norm_future"]) * future_results["climate_exposure_future_min"]
        + 0.3 * future_results["sigma_norm_future"]
    )
    future_results["vulnerability_future_hi"] = (
        (1 - future_results["diversity_norm_future"]) * future_results["climate_exposure_future_max"]
        + 0.3 * future_results["sigma_norm_future"]
    )

    future_results.to_csv(PROC / "vulnerability_scores_future.csv", index=False)
    print(f"\nSaved: {PROC / 'vulnerability_scores_future.csv'}")

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("CURRENT vs FUTURE (ENSEMBLE) COMPARISON")
    print("=" * 60)
    print(f"  Step 5 ensemble exposure: {current_results['climate_exposure'].mean():.4f}")
    print(f"  Future ensemble mean:   {ensemble_mean.mean():.4f} ± {ensemble_std.mean():.4f}")
    print(f"  Exposure change:        +{ensemble_mean.mean() - current_results['climate_exposure'].mean():.4f}")
    print(f"  Current mean diversity: {current_results['diversity_norm'].mean():.4f}")
    print(f"  Future mean diversity:  {future_results['diversity_norm_future'].mean():.4f}")
    print(f"  Diversity change:       {future_results['diversity_norm_future'].mean() - current_results['diversity_norm'].mean():+.4f}")

    print(f"\n  Resilience classification:")
    for cls in ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]:
        pct_curr = (current_results["resilience_class"] == cls).mean() * 100
        pct_fut = (future_results["resilience_class_future"] == cls).mean() * 100
        print(f"    {cls:25s}: current={pct_curr:5.1f}%  future={pct_fut:5.1f}%  "
              f"change={pct_fut-pct_curr:+.1f}%")

    # ═══════════════════════════════════════════════════════
    # Figure 1: Ecosystem resilience — future ensemble
    # ═══════════════════════════════════════════════════════
    print("\nGenerating figures...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    import geopandas as gpd
    import ca_basemap as _cabm
    from pyproj import Transformer

    lat_bin_edges = np.linspace(coast_lats.min(), coast_lats.max(), 76)
    lat_bin_centers = (lat_bin_edges[:-1] + lat_bin_edges[1:]) / 2

    eco_stats = []
    for i in range(len(lat_bin_edges) - 1):
        lo, hi = lat_bin_edges[i], lat_bin_edges[i + 1]
        mask = (future_results["Latitude"] >= lo) & (future_results["Latitude"] < hi)
        pts = future_results[mask]
        if len(pts) == 0:
            continue

        n_species = pts["species"].nunique()
        n_total = len(pts)
        lon_center = np.interp(lat_bin_centers[i], coast_lats, coast_lons)

        eco_stats.append({
            "lat_center": lat_bin_centers[i],
            "lon_center": lon_center,
            "n_species": n_species,
            "mean_vulnerability": pts["vulnerability_future"].mean(),
            "mean_vulnerability_lo": pts["vulnerability_future_lo"].mean(),
            "mean_vulnerability_hi": pts["vulnerability_future_hi"].mean(),
            "mean_diversity": pts["diversity_norm_future"].mean(),
            "mean_exposure": pts["climate_exposure_future"].mean(),
            "mean_exposure_std": pts["climate_exposure_future_std"].mean(),
            "mean_uncertainty": pts["sigma_norm_future"].mean(),
            "pct_critical": (pts["resilience_class_future"] == "Critical").sum() / n_total,
            "pct_at_risk": (pts["resilience_class_future"] == "At Risk").sum() / n_total,
            "pct_resilient": (pts["resilience_class_future"] == "Resilient").sum() / n_total,
            "pct_latent": (pts["resilience_class_future"] == "Latent Vulnerability").sum() / n_total,
        })

    eco_df = pd.DataFrame(eco_stats)

    CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
    CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ca_xmin, ca_ymin = transformer.transform(CA_LON_MIN, CA_LAT_MIN)
    ca_xmax, ca_ymax = transformer.transform(CA_LON_MAX, CA_LAT_MAX)

    fig = plt.figure(figsize=(28, 24))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.25, wspace=0.2)

    # Panel A: Map
    ax_map = fig.add_subplot(gs[:, 0])
    _cabm.add_land(ax_map, ca_xmin, ca_xmax, ca_ymin, ca_ymax)

    vuln_cmap = LinearSegmentedColormap.from_list(
        "vuln", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    )
    eco_gdf = gpd.GeoDataFrame(
        eco_df,
        geometry=gpd.points_from_xy(eco_df["lon_center"], eco_df["lat_center"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    sizes = eco_df["n_species"] / eco_df["n_species"].max() * 200 + 50
    sc = ax_map.scatter(
        eco_gdf.geometry.x, eco_gdf.geometry.y,
        c=eco_df["mean_vulnerability"], cmap=vuln_cmap,
        s=sizes, alpha=0.85,
        edgecolors="black", linewidth=0.4,
        vmin=0.1, vmax=0.7, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax_map, shrink=0.5, pad=0.02)
    cb.set_label("Mean vulnerability (projected diversity × Δ-climate exposure)", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    ax_map.set_xlim(ca_xmin, ca_xmax)
    ax_map.set_ylim(ca_ymin, ca_ymax)
    ax_map.set_title("Ecosystem Resilience to Projected Climate CHANGE\n(5-GCM Ensemble Mean)",
                     fontsize=16, fontweight="bold")
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    for ns, label in [(5, "5 spp"), (15, "15 spp"), (25, "25 spp")]:
        s = ns / eco_df["n_species"].max() * 200 + 50
        ax_map.scatter([], [], s=s, c="gray", alpha=0.6, edgecolors="black",
                       linewidth=0.4, label=label)
    ax_map.legend(title="Species coverage", loc="lower left", fontsize=11,
                  title_fontsize=12, frameon=True, fancybox=True)

    # Panel B: Stacked area
    ax_stack = fig.add_subplot(gs[0, 1])
    ax_stack.fill_between(eco_df["lat_center"], 0, eco_df["pct_critical"],
                          color="#e74c3c", alpha=0.8, label="Critical")
    ax_stack.fill_between(eco_df["lat_center"],
                          eco_df["pct_critical"],
                          eco_df["pct_critical"] + eco_df["pct_latent"],
                          color="#e67e22", alpha=0.8, label="Latent Vulnerability")
    ax_stack.fill_between(eco_df["lat_center"],
                          eco_df["pct_critical"] + eco_df["pct_latent"],
                          eco_df["pct_critical"] + eco_df["pct_latent"] + eco_df["pct_at_risk"],
                          color="#f1c40f", alpha=0.8, label="At Risk")
    ax_stack.fill_between(eco_df["lat_center"],
                          eco_df["pct_critical"] + eco_df["pct_latent"] + eco_df["pct_at_risk"],
                          1.0,
                          color="#2ecc71", alpha=0.8, label="Resilient")

    ax_count = ax_stack.twinx()
    ax_count.plot(eco_df["lat_center"], eco_df["n_species"], "k-", linewidth=2, alpha=0.6)
    ax_count.set_ylabel("Number of species", fontsize=12, color="black")

    ax_stack.set_xlabel("Latitude", fontsize=12)
    ax_stack.set_ylabel("Proportion of species", fontsize=12)
    ax_stack.set_title("Resilience Class Composition\n(Δ-climate exposure, ensemble mean)",
                       fontsize=14, fontweight="bold")
    ax_stack.set_ylim([0, 1])
    ax_stack.legend(loc="upper right", fontsize=10, frameon=True)
    ax_stack.grid(True, alpha=0.2, axis="x")

    # Panel C: Vulnerability components WITH inter-model bands
    ax_comp = fig.add_subplot(gs[1, 1])

    ax_comp.plot(eco_df["lat_center"], 1 - eco_df["mean_diversity"],
                 "b-o", markersize=3, linewidth=2, alpha=0.7,
                 label="Diversity deficit (1 - div_norm)")

    # Exposure: ensemble mean + inter-model band
    ax_comp.plot(eco_df["lat_center"], eco_df["mean_exposure"],
                 "r-s", markersize=3, linewidth=2, alpha=0.7,
                 label="Δ-climate exposure (ensemble mean)")
    ax_comp.fill_between(eco_df["lat_center"],
                         eco_df["mean_exposure"] - eco_df["mean_exposure_std"],
                         eco_df["mean_exposure"] + eco_df["mean_exposure_std"],
                         color="red", alpha=0.15, label="Inter-model ±1 SD")

    ax_comp.plot(eco_df["lat_center"], eco_df["mean_uncertainty"],
                 "gray", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="Model uncertainty")

    # Vulnerability: ensemble mean + min-max band
    ax_comp.plot(eco_df["lat_center"], eco_df["mean_vulnerability"],
                 "k-", linewidth=2.5, alpha=0.9,
                 label="Vulnerability (ensemble mean)")
    ax_comp.fill_between(eco_df["lat_center"],
                         eco_df["mean_vulnerability_lo"],
                         eco_df["mean_vulnerability_hi"],
                         color="black", alpha=0.1, label="Vulnerability (GCM range)")

    ax_comp.set_xlabel("Latitude", fontsize=12)
    ax_comp.set_ylabel("Score (0-1)", fontsize=12)
    ax_comp.set_title("Vulnerability Components\n(Δ-climate exposure, inter-model uncertainty)",
                      fontsize=14, fontweight="bold")
    ax_comp.legend(fontsize=9, loc="best", frameon=True)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.set_ylim([0, 1])

    gcm_str = ", ".join(GCMS)
    fig.suptitle(
        f"Ecosystem-Level Vulnerability to Climate CHANGE — California Coast\n"
        f"SSP5-8.5 {PERIOD} | 5-GCM Ensemble ({gcm_str})\n"
        f"Marine: Bio-ORACLE avg(2080, 2090) | "
        f"{future_results['species'].nunique()} species, {len(future_results)} predictions",
        fontsize=17, fontweight="bold", y=0.98,
    )
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"ecosystem_resilience_future.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'ecosystem_resilience_future.png'}")

    # ═══════════════════════════════════════════════════════
    # Figure 2: Diversity vs Exposure — 3-panel (He only)
    # ═══════════════════════════════════════════════════════
    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    HE_REF_LO = global_df["gen_div"].quantile(0.02)
    HE_REF_HI = global_df["gen_div"].quantile(0.98)
    global_df["diversity_norm"] = (
        (global_df["gen_div"] - HE_REF_LO) / (HE_REF_HI - HE_REF_LO)
    ).clip(0, 1)
    global_df["climate_exposure"] = compute_climate_exposure(global_df)

    ca_he_current = current_results[current_results["metric_type"] == "He"]
    ca_he_future = future_results[future_results["metric_type"] == "He"]
    he_species = sorted(ca_he_current["species"].unique())
    species_colors = plt.cm.tab10(np.linspace(0, 1, len(he_species)))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: Global He historical
    ax = axes[0]
    hb = ax.hexbin(global_df["climate_exposure"], global_df["diversity_norm"],
                   gridsize=30, cmap="YlOrRd", mincnt=1)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Δ-climate exposure (projected change)", fontsize=11)
    ax.set_ylabel("He (global He ref)", fontsize=11)
    ax.set_title(f"Global He (Historical)\n({len(global_df):,} pops, "
                 f"{global_df['species'].nunique()} spp.)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.colorbar(hb, ax=ax, label="Population count", shrink=0.8)
    for txt, x, y, c in [("Resilient", 0.25, 0.75, "#2ecc71"),
                           ("At Risk", 0.75, 0.75, "#f1c40f"),
                           ("Latent\nVulnerable", 0.25, 0.25, "#e67e22"),
                           ("Critical", 0.75, 0.25, "#e74c3c")]:
        ax.text(x, y, txt, ha="center", fontsize=10, color=c, alpha=0.4, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel 2: CA He current on global background
    ax = axes[1]
    ax.hexbin(global_df["climate_exposure"], global_df["diversity_norm"],
              gridsize=30, cmap="Blues", mincnt=1, alpha=0.35)
    for i, sp in enumerate(he_species):
        sp_d = ca_he_current[ca_he_current["species"] == sp]
        ax.scatter(sp_d["climate_exposure"], sp_d["diversity_norm"],
                   s=40, alpha=0.7, color=species_colors[i],
                   edgecolors="black", linewidth=0.4, label=sp, zorder=3)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Climate Exposure (historical)", fontsize=11)
    ax.set_ylabel("He (global He ref)", fontsize=11)
    ax.set_title(f"California He (Historical)\n({len(ca_he_current)} pops, {len(he_species)} spp.)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(fontsize=7, ncol=2, loc="lower left", frameon=True, framealpha=0.9)
    for txt, x, y, c in [("Resilient", 0.25, 0.75, "#2ecc71"),
                           ("At Risk", 0.75, 0.75, "#f1c40f"),
                           ("Latent\nVulnerable", 0.25, 0.25, "#e67e22"),
                           ("Critical", 0.75, 0.25, "#e74c3c")]:
        ax.text(x, y, txt, ha="center", fontsize=10, color=c, alpha=0.4, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Panel 3: CA He FUTURE (ensemble) on global background + error bars
    ax = axes[2]
    ax.hexbin(global_df["climate_exposure"], global_df["diversity_norm"],
              gridsize=30, cmap="Blues", mincnt=1, alpha=0.35)
    for i, sp in enumerate(he_species):
        sp_d = ca_he_future[ca_he_future["species"] == sp]
        # Horizontal error bars showing inter-model spread
        ax.errorbar(sp_d["climate_exposure_future"], sp_d["diversity_norm"],
                    xerr=sp_d["climate_exposure_future_std"],
                    fmt="none", ecolor=species_colors[i], alpha=0.2, zorder=2)
        ax.scatter(sp_d["climate_exposure_future"], sp_d["diversity_norm"],
                   s=40, alpha=0.8, color=species_colors[i],
                   edgecolors="black", linewidth=0.5, label=sp, zorder=3)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Δ-climate exposure (SSP5-8.5 ensemble ± SD)", fontsize=11)
    ax.set_ylabel("He (global He ref)", fontsize=11)
    ax.set_title(f"California He Future (5-GCM Ensemble)\nvs Global Historical\n"
                 f"(blue bg = global historical)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=True, framealpha=0.9)
    for txt, x, y, c in [("Resilient", 0.25, 0.75, "#2ecc71"),
                           ("At Risk", 0.75, 0.75, "#f1c40f"),
                           ("Latent\nVulnerable", 0.25, 0.25, "#e67e22"),
                           ("Critical", 0.75, 0.25, "#e74c3c")]:
        ax.text(x, y, txt, ha="center", fontsize=10, color=c, alpha=0.4, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.suptitle(
        f"Genetic Diversity vs Climate Exposure — He Species\n"
        f"Marine-Weighted Exposure + Global He Normalization | "
        f"Future: SSP5-8.5 {PERIOD} (5-GCM Ensemble)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"div_vs_exposure_future.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'div_vs_exposure_future.png'}")

    print("\n" + "=" * 60)
    print("DONE — Multi-model ensemble complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

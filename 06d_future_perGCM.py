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

Framing (decision B): exposure is the coming (present→2100) change, so
vulnerability/resilience is well-posed for the PRESENT day only and is quantified
per GCM (it varies through the exposure term). The FUTURE is reported as projected
diversity + between-GCM uncertainty, not a vulnerability class, because a
forward-looking post-2100 exposure is undefined (CMIP6 ends at 2100).

Outputs:
  data/processed/future_perGCM_diversity_exposure.csv     (intermediate inputs)
  figures/resilience_agreement_perGCM.{pdf,png}           (present-day, per GCM)
  data/processed/resilience_agreement_perGCM.csv
  figures/future_diversity_perGCM_uncertainty.{pdf,png}   (future projected diversity)
  data/processed/future_diversity_perGCM_uncertainty.csv
"""

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
import os
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

    # Per-GCM 1 km future diversity (from 11_gcm_ensemble_diversity.py) replaces the
    # GCM-averaged 10 km single prediction; row-aligned with vulnerability_scores.csv.
    gcm_div = pd.read_csv(PROC / "gcm_ensemble_diversity.csv")
    if not (len(gcm_div) == len(current_results)
            and (gcm_div["species"].values == current_results["species"].values).all()):
        raise SystemExit("gcm_ensemble_diversity.csv not row-aligned with "
                         "vulnerability_scores.csv; re-run 11_gcm_ensemble_diversity.py")
    future_pred_mu = gcm_div["future_mu_mean"].values.astype(float)
    future_pred_sigma = gcm_div["future_mu_sd"].values.astype(float)  # between-GCM SD
    for sp in current_results["species"].unique():
        m = (current_results["species"] == sp).values
        print(f"  {sp}: current He/pi={current_results.loc[m,'pred_mu'].mean():.4f} → "
              f"future={future_pred_mu[m].mean():.4f} "
              f"({future_pred_mu[m].mean() - current_results.loc[m,'pred_mu'].mean():+.4f})")

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
    # Per-GCM future diversity (cross-species reference), carried through from
    # gcm_ensemble_diversity.csv for the global-context figure (Figure 2).
    for gcm in GCMS:
        col = f"future_dnorm_{gcm}"
        if col in gcm_div.columns:
            future_results[col] = gcm_div[col].values

    # ── Align future scoring to the present-day (step 05) definition ──
    #   Diversity: within-species normalization on the joint present∪future scale
    #   (diversity_scaling.csv, written by 06c) — same axis as the present-day score.
    #   Stressor: LULC-augmented total stress = (1-w)·Δ-climate exposure + w·LULC
    #   pressure, LULC held at present-day (so it does not vary across GCMs).
    LULC_W = getattr(_r, "LULC_WEIGHT", 0.25)
    _scale_path = PROC / "diversity_scaling.csv"
    _joint = (pd.read_csv(_scale_path).set_index("species")
              if _scale_path.exists() else None)
    print("  future diversity scale: " + ("present ∪ future (from 06c)"
          if _joint is not None else "per-GCM spatial range (bootstrap — rerun after 06c)"))
    _sp = future_results["species"].values
    _lulc_press = pd.to_numeric(future_results["lulc_pressure"], errors="coerce").values
    _lulc_have = (future_results["lulc_available"].fillna(False).astype(bool).values
                  & np.isfinite(_lulc_press))

    def _within_norm(raw_mu):
        raw_mu = np.asarray(raw_mu, float)
        out = np.full(len(raw_mu), 0.5)
        for sp in np.unique(_sp):
            m = _sp == sp
            if _joint is not None and sp in _joint.index:
                lo, hi = _joint.loc[sp, "div_lo"], _joint.loc[sp, "div_hi"]
            else:
                v = raw_mu[m]; lo, hi = np.nanmin(v), np.nanmax(v)
            if hi - lo > 1e-8:
                out[m] = np.clip((raw_mu[m] - lo) / (hi - lo), 0, 1)
        return out

    def _total_stress(exposure):
        exposure = np.asarray(exposure, float)
        ts = exposure.copy()
        ts[_lulc_have] = ((1 - LULC_W) * exposure[_lulc_have]
                          + LULC_W * _lulc_press[_lulc_have])
        return ts

    future_results["diversity_norm_within_future"] = _within_norm(future_pred_mu)
    future_results["total_stress_future"] = _total_stress(ensemble_mean)

    # Per-GCM within-species diversity + total stress (for classification and bands)
    within_gcm_cols = []
    for gcm in GCMS:
        wcol = f"future_dnorm_within_{gcm}"
        future_results[wcol] = _within_norm(gcm_div[f"future_mu_{gcm}"].values)
        within_gcm_cols.append(wcol)
        future_results[f"total_stress_{gcm}"] = _total_stress(
            future_results[f"exposure_{gcm}"].values)

    # NOTE: No future vulnerability/resilience is computed. Under decision B the
    # future is reported as projected diversity only (see _future_diversity_
    # uncertainty); a future vulnerability would require an undefined post-2100
    # exposure. This table holds the per-GCM projected-diversity and exposure
    # inputs for reproducibility.
    future_results.to_csv(PROC / "future_perGCM_diversity_exposure.csv", index=False)
    print(f"\nSaved: {PROC / 'future_perGCM_diversity_exposure.csv'}")

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("CURRENT vs FUTURE (ENSEMBLE) COMPARISON")
    print("=" * 60)
    print(f"  Step 5 ensemble exposure: {current_results['climate_exposure'].mean():.4f}")
    print(f"  Future ensemble mean:   {ensemble_mean.mean():.4f} ± {ensemble_std.mean():.4f}")
    print(f"  Exposure change:        +{ensemble_mean.mean() - current_results['climate_exposure'].mean():.4f}")
    _cur_dw = current_results["diversity_norm_within"] if "diversity_norm_within" in current_results \
        else current_results["diversity_norm"]
    print(f"  Current mean diversity (within-sp): {_cur_dw.mean():.4f}")
    print(f"  Future mean diversity (within-sp):  {future_results['diversity_norm_within_future'].mean():.4f}")
    print(f"  Diversity change:       {future_results['diversity_norm_within_future'].mean() - _cur_dw.mean():+.4f}")

    # ── Figure setup (shared by the per-GCM figures below) ──
    print("\nGenerating figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import geopandas as gpd
    import ca_basemap as _cabm
    from pyproj import Transformer

    CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
    CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ca_xmin, ca_ymin = transformer.transform(CA_LON_MIN, CA_LAT_MIN)
    ca_xmax, ca_ymax = transformer.transform(CA_LON_MAX, CA_LAT_MAX)

    # Decision B: present-day is the sole quantified vulnerability; the future is
    # reported as projected diversity, not vulnerability. The former future-
    # vulnerability figures (ecosystem_resilience_future_perGCM, div_vs_exposure_
    # future_perGCM) are intentionally no longer generated.

    # ═══════════════════════════════════════════════════════
    # Figures 3 & 4: Per-GCM resilience classification — model agreement
    #   Each GCM is classified INDEPENDENTLY with the present-day rule
    #   (classify_resilience). Exposure is the projected climate CHANGE
    #   (SSP5-8.5) and varies by GCM, so BOTH epochs carry GCM spread; the
    #   two epochs differ only in the diversity term (present observed range
    #   vs projected end-of-century). We report, per population, the modal
    #   class, the fraction of models backing it (classification
    #   uncertainty), and the per-model probability of each class.
    # ═══════════════════════════════════════════════════════
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    CLASSES = ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]
    CLASS_COL = {"Resilient": "#2ecc71", "At Risk": "#f1c40f",
                 "Latent Vulnerability": "#e67e22", "Critical": "#e74c3c"}
    geom = future_results[["species", "metric_type", "Latitude", "Longitude"]].reset_index(drop=True)

    def _agreement_analysis(div_by_gcm, stress_by_gcm, sigma, epoch_label, fname_stub, suptitle):
        """Classify per GCM, summarize agreement, draw the 3-panel figure, write CSV."""
        per = {g: classify_resilience(np.asarray(div_by_gcm[g], float),
                                      np.asarray(stress_by_gcm[g], float),
                                      np.asarray(sigma, float)) for g in GCMS}
        Lc = pd.DataFrame(per)
        vote = pd.DataFrame({c: (Lc == c).sum(axis=1) for c in CLASSES})
        out = geom.copy()
        out["modal_class_perGCM"] = vote.idxmax(axis=1).values
        out["model_agreement"] = (vote.max(axis=1) / len(GCMS)).values
        out["p_critical_perGCM"] = (vote["Critical"] / len(GCMS)).values
        n_pop = len(out)
        agree_n = (out["model_agreement"] * len(GCMS)).round().astype(int)

        print("\n" + "=" * 60)
        print(f"PER-GCM RESILIENCE CLASSIFICATION — {epoch_label} (model agreement)")
        print("=" * 60)
        print("Modal-class distribution:")
        for c in CLASSES:
            k = (out["modal_class_perGCM"] == c).sum()
            print(f"    {c:22s}: {k:5d}  ({k/n_pop*100:5.1f}%)")
        print(f"  Unanimous (5/5): {(agree_n==5).mean()*100:5.1f}%   "
              f"Majority (>=3/5): {(agree_n>=3).mean()*100:5.1f}%   "
              f"No majority (<3/5): {(agree_n<3).mean()*100:5.1f}%")

        fig = plt.figure(figsize=(20, 13))
        gs3 = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.28, wspace=0.18)

        # Panel A: model-agreement map (modal class; opacity = agreement)
        axm = fig.add_subplot(gs3[:, 0])
        _cabm.add_land(axm, ca_xmin, ca_xmax, ca_ymin, ca_ymax)
        pts = gpd.GeoDataFrame(
            out, geometry=gpd.points_from_xy(out["Longitude"], out["Latitude"]),
            crs="EPSG:4326").to_crs(epsg=3857)
        alpha = np.interp(out["model_agreement"], [0.4, 1.0], [0.25, 0.95])
        face = [to_rgba(CLASS_COL[c], a) for c, a in zip(out["modal_class_perGCM"], alpha)]
        axm.scatter(pts.geometry.x, pts.geometry.y, c=face, s=90, edgecolors="none", zorder=3)
        contested = (out["model_agreement"] < 0.6).values
        axm.scatter(pts.geometry.x[contested], pts.geometry.y[contested],
                    facecolors="none", edgecolors="black", s=200, linewidth=1.1, zorder=4)
        axm.set_xlim(ca_xmin, ca_xmax); axm.set_ylim(ca_ymin, ca_ymax)
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title("Per-GCM resilience classification — model-agreement map\n"
                      "(color = modal class across 5 GCMs; opacity = agreement)",
                      fontsize=15, fontweight="bold")
        leg1 = [Patch(facecolor=CLASS_COL[c], label=c) for c in CLASSES]
        leg1.append(Line2D([], [], marker="o", color="black", mfc="none", ls="none",
                           ms=11, label="No majority (<3/5)"))
        axm.legend(handles=leg1, loc="lower left", fontsize=10, frameon=True, framealpha=0.95)

        # Panel B: per-latitude mean class-probability (stacked area)
        axb = fig.add_subplot(gs3[0, 1])
        edges = np.linspace(CA_LAT_MIN, CA_LAT_MAX, 40)
        ctr = (edges[:-1] + edges[1:]) / 2
        prob = {c: [] for c in CLASSES}; lat_keep = []
        for i in range(len(edges) - 1):
            m = ((out["Latitude"] >= edges[i]) & (out["Latitude"] < edges[i + 1])).values
            if m.sum() == 0:
                continue
            lat_keep.append(ctr[i])
            sub = vote[m] / len(GCMS)
            for c in CLASSES:
                prob[c].append(sub[c].mean())
        lat_keep = np.array(lat_keep); base = np.zeros_like(lat_keep)
        for c in CLASSES:
            vals = np.array(prob[c])
            axb.fill_between(lat_keep, base, base + vals, color=CLASS_COL[c], alpha=0.85, label=c)
            base = base + vals
        axb.set_xlim(CA_LAT_MIN, CA_LAT_MAX); axb.set_ylim(0, 1)
        axb.set_xlabel("Latitude", fontsize=12)
        axb.set_ylabel("Mean per-model class probability", fontsize=12)
        axb.set_title("Classification uncertainty by latitude\n"
                      "(average across 5 GCMs of the class vote share)",
                      fontsize=13, fontweight="bold")
        axb.legend(loc="upper right", fontsize=9, frameon=True, ncol=2)
        axb.grid(True, alpha=0.2, axis="x")

        # Panel C: model consensus strength
        axc = fig.add_subplot(gs3[1, 1])
        order = [5, 4, 3, 2]
        lab_txt = {5: "Unanimous (5/5)", 4: "Strong (4/5)", 3: "Majority (3/5)", 2: "Split (≤2/5)"}
        lvl = {lv: out[agree_n == lv]["modal_class_perGCM"].value_counts() for lv in order}
        xpos = np.arange(len(order)); bottom = np.zeros(len(order))
        for c in CLASSES:
            h = [lvl[lv].get(c, 0) / n_pop * 100 for lv in order]
            axc.bar(xpos, h, bottom=bottom, color=CLASS_COL[c], label=c, width=0.7)
            bottom += h
        axc.set_xticks(xpos); axc.set_xticklabels([lab_txt[lv] for lv in order], fontsize=10)
        axc.set_ylabel("% of populations", fontsize=12)
        axc.set_title("Model consensus strength\n(populations by agreement level, colored by modal class)",
                      fontsize=13, fontweight="bold")
        axc.grid(True, alpha=0.2, axis="y")
        for i, lv in enumerate(order):
            axc.text(xpos[i], bottom[i] + 1.5, f"{(agree_n==lv).mean()*100:.0f}%",
                     ha="center", fontsize=10, fontweight="bold")

        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.99)
        for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
            plt.savefig(FIG_DIR / f"{fname_stub}.{_ext}", bbox_inches="tight", **_kw)
        plt.close()
        print(f"Saved: {FIG_DIR / (fname_stub + '.png')}")
        out.to_csv(PROC / f"{fname_stub}.csv", index=False)
        print(f"Saved: {PROC / (fname_stub + '.csv')}")
        return out

    def _future_diversity_uncertainty(div_by_gcm, present_div, fname_stub, suptitle):
        """Projected end-of-century diversity with per-GCM uncertainty.

        The future is reported as projected genetic diversity, NOT as a
        vulnerability class: Δ-climate exposure is the present→2100 change, so a
        forward-looking (post-2100) exposure — which the 2100 state would need
        for a well-posed future vulnerability — is undefined. We therefore show
        the ensemble-mean projected diversity and the between-GCM spread.
        """
        Mx = np.column_stack([np.asarray(div_by_gcm[g], float) for g in GCMS])
        mu, sd = Mx.mean(axis=1), Mx.std(axis=1)
        out = geom.copy()
        out["present_div"] = np.asarray(present_div, float)
        out["future_div_mean"] = mu
        out["future_div_sd"] = sd
        out["future_div_change"] = mu - out["present_div"]

        print("\n" + "=" * 60)
        print("PER-GCM PROJECTED DIVERSITY (future, within-species)")
        print("=" * 60)
        print(f"  present mean: {out['present_div'].mean():.4f}   "
              f"future mean: {mu.mean():.4f}   change: {out['future_div_change'].mean():+.4f}")
        print(f"  between-GCM SD: mean={sd.mean():.4f}  max={sd.max():.4f}")

        fig = plt.figure(figsize=(20, 13))
        gsd = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.28, wspace=0.2)

        # Panel A: map, color = ensemble-mean projected diversity, opacity = certainty
        axm = fig.add_subplot(gsd[:, 0])
        _cabm.add_land(axm, ca_xmin, ca_xmax, ca_ymin, ca_ymax)
        pts = gpd.GeoDataFrame(
            out, geometry=gpd.points_from_xy(out["Longitude"], out["Latitude"]),
            crs="EPSG:4326").to_crs(epsg=3857)
        sd_cap = np.nanpercentile(sd, 90) or 1e-6
        alpha = np.interp(sd, [0.0, sd_cap], [0.95, 0.3])
        cmap = plt.cm.viridis
        rgba = cmap(mu)
        rgba[:, 3] = alpha
        sc = axm.scatter(pts.geometry.x, pts.geometry.y, c=rgba, s=90, edgecolors="none", zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
        cb = plt.colorbar(sm, ax=axm, shrink=0.5, pad=0.02)
        cb.set_label("Projected within-species diversity (ensemble mean)", fontsize=12)
        axm.set_xlim(ca_xmin, ca_xmax); axm.set_ylim(ca_ymin, ca_ymax)
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title("Projected diversity — model-uncertainty map\n"
                      "(color = ensemble mean; opacity = certainty)",
                      fontsize=15, fontweight="bold")

        # Panel B: per-latitude present vs future diversity + inter-GCM band
        axb = fig.add_subplot(gsd[0, 1])
        edges = np.linspace(CA_LAT_MIN, CA_LAT_MAX, 40)
        ctr = (edges[:-1] + edges[1:]) / 2
        lat_k, pres_p, fut_p, band = [], [], [], []
        for i in range(len(edges) - 1):
            m = ((out["Latitude"] >= edges[i]) & (out["Latitude"] < edges[i + 1])).values
            if m.sum() == 0:
                continue
            lat_k.append(ctr[i]); pres_p.append(out["present_div"][m].mean())
            fut_p.append(mu[m].mean())
            band.append(np.std([np.asarray(div_by_gcm[g], float)[m].mean() for g in GCMS]))
        lat_k = np.array(lat_k); fut_p = np.array(fut_p); band = np.array(band)
        axb.plot(lat_k, pres_p, "k--", lw=2, alpha=0.7, label="Present day")
        axb.plot(lat_k, fut_p, "b-o", ms=3, lw=2, alpha=0.8, label="Projected 2100 (ensemble)")
        axb.fill_between(lat_k, fut_p - band, fut_p + band, color="blue", alpha=0.15,
                         label="Between-GCM ±1 SD")
        axb.set_xlim(CA_LAT_MIN, CA_LAT_MAX); axb.set_ylim(0, 1)
        axb.set_xlabel("Latitude", fontsize=12)
        axb.set_ylabel("Within-species diversity", fontsize=12)
        axb.set_title("Projected diversity by latitude\n(present vs 2100, inter-model band)",
                      fontsize=13, fontweight="bold")
        axb.legend(loc="upper right", fontsize=9, frameon=True)
        axb.grid(True, alpha=0.2)

        # Panel C: between-GCM projection uncertainty by latitude
        axc = fig.add_subplot(gsd[1, 1])
        sd_p = []
        for i in range(len(edges) - 1):
            m = ((out["Latitude"] >= edges[i]) & (out["Latitude"] < edges[i + 1])).values
            if m.sum() == 0:
                continue
            sd_p.append(sd[m].mean())
        axc.fill_between(lat_k, 0, np.array(sd_p), color="#8e44ad", alpha=0.6)
        axc.set_xlim(CA_LAT_MIN, CA_LAT_MAX)
        axc.set_xlabel("Latitude", fontsize=12)
        axc.set_ylabel("Between-GCM SD of projected diversity", fontsize=12)
        axc.set_title("Projection uncertainty by latitude\n(disagreement among the 5 GCMs)",
                      fontsize=13, fontweight="bold")
        axc.grid(True, alpha=0.2)

        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.99)
        for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
            plt.savefig(FIG_DIR / f"{fname_stub}.{_ext}", bbox_inches="tight", **_kw)
        plt.close()
        print(f"Saved: {FIG_DIR / (fname_stub + '.png')}")
        out.to_csv(PROC / f"{fname_stub}.csv", index=False)
        print(f"Saved: {PROC / (fname_stub + '.csv')}")
        return out

    # Shared per-GCM stressor (Δ-climate exposure varies by GCM; LULC held at present)
    stress_by_gcm = {g: future_results[f"total_stress_{g}"].values for g in GCMS}

    # ── VULNERABILITY / RESILIENCE: present-day only, quantified per GCM ──
    # Δ-climate exposure is the coming (present→2100) change, so the present-day
    # classification is well-posed and carries GCM spread through that exposure.
    # The future is NOT classified for vulnerability (see _future_diversity_
    # uncertainty): its forward-looking post-2100 exposure is undefined.
    _pres_sigma = (current_results["sigma_norm"].values
                   if "sigma_norm" in current_results else future_results["sigma_norm_future"].values)
    pres_div_fixed = current_results["diversity_norm_within"].values
    pres_div = {g: pres_div_fixed for g in GCMS}
    present_agree = _agreement_analysis(
        pres_div, stress_by_gcm, _pres_sigma,
        "PRESENT DAY", "resilience_agreement_perGCM",
        f"Present-day genetic resilience quantified per GCM — California Coast\n"
        f"Δ-climate exposure (coming SSP5-8.5 change to {PERIOD}) varies by GCM; "
        f"present-day diversity | each of {len(GCMS)} GCMs classified independently")

    # ── FUTURE: projected diversity + per-GCM uncertainty (no vuln classification) ──
    fut_div = {g: future_results[f"future_dnorm_within_{g}"].values for g in GCMS}
    _future_diversity_uncertainty(
        fut_div, pres_div_fixed, "future_diversity_perGCM_uncertainty",
        f"Projected end-of-century genetic diversity per GCM — California Coast\n"
        f"SSP5-8.5 {PERIOD} | within-species diversity, ensemble mean and between-GCM spread")

    print("\n" + "=" * 60)
    print("DONE — Multi-model ensemble complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Climate exposure = the magnitude of PROJECTED CHANGE in climate parameters
between the historical baseline and a climate change scenario.

This module is the single definition of exposure for the whole pipeline. Every
script that scores exposure calls `compute_climate_exposure()` from here, so
exposure is never the absolute value of a climate variable in one time slice — a
site is exposed because its conditions shift, not because it is already warm.

  historical baseline           climate change scenario
  ------------------           -----------------------
  WorldClim 1970-2000    ->    WorldClim CMIP6 SSP5-8.5 2081-2100 (per GCM)
  Bio-ORACLE @2020       ->    Bio-ORACLE SSP5-8.5 avg(2080, 2090)

The marine baseline is the 2020 step of the same SSP5-8.5 product as the
scenario, not the observational 2000-2019 climatology, so the change carries no
model-vs-observation bias.

Components (each normalized against a fixed reference CHANGE range so scores are
absolute, comparable between regions and scenarios, not min-max within a sample):

  Marine (70%)                          Terrestrial (30%)
    0.10  d sst_mean    warming           0.10  d bio5   hottest-month warming
    0.10  d sst_max     warming           0.08  d bio1   mean annual warming
    0.08  |d sst_range| seasonal swing    0.06  |d bio4| seasonality change
    0.15  -d ph_mean    acidification     0.06  bio14 drying (relative loss)
    0.12  -d o2_mean    deoxygenation
    0.08  d sst / baseline sst_range      thermal novelty

Where marine data are unavailable the four terrestrial terms are renormalized to
sum to 1.

Usage:
    import climate_delta
    exposure = climate_delta.compute_climate_exposure(df)              # GCM ensemble
    exposure = climate_delta.compute_climate_exposure(df, gcm="MIROC6")

`df` needs Latitude and Longitude columns; every other column is ignored, since
exposure no longer depends on the present-day state of the site.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from scipy.interpolate import NearestNDInterpolator

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
RASTER_DIR = ROOT / "data" / "env_rasters"
CACHE_PATH = PROC / "climate_delta_cache.json"

PERIOD = "2081-2100"
SCENARIO = "ssp585"
GCMS = ["BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR", "MIROC6"]

BASELINE_TIME = "2020-01-01T00:00:00Z"
SCENARIO_TIMES = ["2080-01-01T00:00:00Z", "2090-01-01T00:00:00Z"]

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"
DATASETS = {
    f"thetao_{SCENARIO}_2020_2100_depthsurf": [
        ("thetao_mean", "sst_mean"), ("thetao_max", "sst_max"),
        ("thetao_range", "sst_range")],
    f"o2_{SCENARIO}_2020_2100_depthsurf": [("o2_mean", "o2_mean")],
    f"ph_{SCENARIO}_2020_2100_depthsurf": [("ph_mean", "ph_mean")],
}
MARINE_VARS = ["sst_mean", "sst_max", "sst_range", "o2_mean", "ph_mean"]

# California coast at native Bio-ORACLE resolution, rest of the world at 1 degree
CA_BOX = (29.0, 43.0, -126.0, -114.0)      # lat_min, lat_max, lon_min, lon_max
GLOBAL_STRIDE = 20                          # 0.05 deg -> 1 deg

BIOS = [1, 4, 5, 14]

# 1 km (30 arc-sec) WorldClim crops for the California window. Where they cover a
# point they are used in place of the 10 arc-min global rasters, which stay in
# service for everything outside the box (e.g. the global population cloud).
FINE_DIR = Path("/Users/liyingnceas/Library/CloudStorage/Box-Box/Genomics/data/WorldClim_1km")
FINE_PRESENT_DIR = FINE_DIR / "present_1970-2000"
FINE_FUTURE_DIR = FINE_DIR / "ssp585_2081-2100"
FINE_BANDS = [1, 4, 5, 6, 7, 12, 13, 14, 15]     # band order inside the crops


def _fine_present_path(bio):
    return FINE_PRESENT_DIR / f"wc2.1_30s_bio_{bio}_CA.tif"


def _fine_scenario_path(gcm):
    return FINE_FUTURE_DIR / f"wc2.1_30s_bioc_{gcm}_ssp585_{PERIOD}_CA.tif"


def _in_ca_box(lats, lons):
    return ((lats >= CA_BOX[0]) & (lats <= CA_BOX[1]) &
            (lons >= CA_BOX[2]) & (lons <= CA_BOX[3]))


def sample_bio(bio, lats, lons, gcm=None):
    """One bioclim variable at each point, at the finest resolution available.

    gcm=None samples the historical baseline; a GCM name samples that model's
    SSP5-8.5 2081-2100 field.
    """
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    out = np.full(len(lats), np.nan)

    fine_path = _fine_present_path(bio) if gcm is None else _fine_scenario_path(gcm)
    fine_band = 1 if gcm is None else (FINE_BANDS.index(bio) + 1 if bio in FINE_BANDS else None)
    use_fine = fine_path.exists() and fine_band is not None
    inside = _in_ca_box(lats, lons) if use_fine else np.zeros(len(lats), bool)

    if inside.any():
        arr = _read_band(fine_path, fine_band)
        rows, cols = _rowcol(fine_path, lats[inside], lons[inside], arr.shape)
        out[inside] = _sample(arr, rows, cols)

    rest = ~inside
    if rest.any():
        path = _present_path(bio) if gcm is None else _scenario_path(gcm)
        band = 1 if gcm is None else bio
        arr = _read_band(path, band)
        rows, cols = _rowcol(path, lats[rest], lons[rest], arr.shape)
        out[rest] = _sample(arr, rows, cols)

    return out


def resolution_note(lats, lons):
    """How many points get the 1 km crops."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    fine = _fine_present_path(1).exists()
    n = int(_in_ca_box(lats, lons).sum()) if fine else 0
    return f"{n}/{len(lats)} points at 30 arc-sec (~1 km), rest at 10 arc-min (~18.5 km)"

# ── Reference CHANGE ranges: the plausible SSP5-8.5 end-of-century envelope ──
DELTA_REF = {
    "d_sst_mean":  (0.0, 4.0),     # degC of ocean warming
    "d_sst_max":   (0.0, 5.0),     # degC
    "d_sst_range": (0.0, 2.0),     # degC, absolute change in annual range
    "d_ph":        (0.0, 0.40),    # pH units of decline
    "d_o2":        (0.0, 20.0),    # mmol/m3 of decline
    "sst_novelty": (0.0, 1.0),     # warming / baseline annual SST range
    "d_bio5":      (0.0, 8.0),     # degC hottest-month warming
    "d_bio1":      (0.0, 7.0),     # degC mean annual warming
    "d_bio4":      (0.0, 150.0),   # sd*100, absolute change in seasonality
    "dry_bio14":   (0.0, 0.50),    # fractional loss of driest-month precipitation
}
WEIGHTS = {
    "d_sst_mean": 0.10, "d_sst_max": 0.10, "d_sst_range": 0.08,
    "d_ph": 0.15, "d_o2": 0.12, "sst_novelty": 0.08,
    "d_bio5": 0.10, "d_bio1": 0.08, "d_bio4": 0.06, "dry_bio14": 0.06,
}
TERRESTRIAL = ["d_bio5", "d_bio1", "d_bio4", "dry_bio14"]
MARINE = [k for k in WEIGHTS if k not in TERRESTRIAL]

_terr_tot = sum(WEIGHTS[k] for k in TERRESTRIAL)
WEIGHTS_TERR_ONLY = {k: WEIGHTS[k] / _terr_tot for k in TERRESTRIAL}


# ═══════════════════════════════════════════════════════════
# Rasters: lazily loaded, kept in memory
# ═══════════════════════════════════════════════════════════
_bands = {}
_grid = {}


def _read_band(path, band):
    """Load a band, with sea cells filled from the nearest land cell.

    A coastal transect sits in the sea, where the terrestrial rasters have no
    data — at 10 arc-min a coastal cell straddles the shore and covers the point,
    but at 30 arc-sec the coastline resolves and most points fall on nodata. Each
    band is therefore filled once with the value of its nearest valid cell, which
    is both what a coastal population experiences and continuous along the shore.
    Filling per-point from a growing window instead makes the value jump wherever
    the window changes size.
    """
    key = (str(path), band)
    if key not in _bands:
        with rasterio.open(path) as src:
            arr = src.read(band).astype("float32")
            arr[(arr < -1e30) | (src.nodata is not None and arr == src.nodata)] = np.nan
            _grid[str(path)] = src.transform
        bad = np.isnan(arr)
        if bad.any() and not bad.all():
            from scipy.ndimage import distance_transform_edt
            _, (ri, ci) = distance_transform_edt(bad, return_indices=True)
            arr = arr[ri, ci]
        _bands[key] = arr
    return _bands[key]


def _rowcol(path, lats, lons, shape):
    t = _grid[str(path)]
    cols = np.floor((np.asarray(lons, float) - t.c) / t.a).astype(int)
    rows = np.floor((np.asarray(lats, float) - t.f) / t.e).astype(int)
    return np.clip(rows, 0, shape[0] - 1), np.clip(cols, 0, shape[1] - 1)


def _sample(arr, rows, cols):
    """Value at each (row, col). Bands are already nearest-filled by _read_band."""
    return arr[rows, cols]


def _present_path(bio):
    return RASTER_DIR / f"wc2.1_10m_bio_{bio}.tif"


def _scenario_path(gcm):
    p = RASTER_DIR / "future_ssp585" / f"wc2.1_10m_bioc_{gcm}_{SCENARIO}_{PERIOD}.tif"
    return p if p.exists() else RASTER_DIR / f"wc2.1_10m_bioc_{gcm}_{SCENARIO}_{PERIOD}.tif"


def terrestrial_change(lats, lons, gcm=None):
    """Baseline -> scenario change in bio1, bio4, bio5, bio14 at each point."""
    gcms = GCMS if gcm is None else [gcm]
    out = {}
    for bio in BIOS:
        pres = sample_bio(bio, lats, lons)
        futs = [sample_bio(bio, lats, lons, gcm=g) for g in gcms]
        out[bio] = (np.mean(np.column_stack(futs), axis=1), pres)
    return out


# ═══════════════════════════════════════════════════════════
# Bio-ORACLE: California box at native resolution, world at 1 degree
# ═══════════════════════════════════════════════════════════
_marine_cache = None
_interp = {}


def _cache():
    global _marine_cache
    if _marine_cache is None:
        _marine_cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    return _marine_cache


def _fetch(dataset, var_list, time_str, box, stride):
    cache = _cache()
    key = f"{dataset}|{time_str}|{box}|{stride}"
    if key in cache:
        return pd.DataFrame(cache[key])
    lat_min, lat_max, lon_min, lon_max = box
    step = f":{stride}:" if stride > 1 else ":"
    var_str = ",".join(
        f"{v}[({time_str})][({lat_min}){step}({lat_max})][({lon_min}){step}({lon_max})]"
        for v, _ in var_list)
    print(f"    Bio-ORACLE: {dataset} @ {time_str[:4]} "
          f"({'global 1deg' if stride > 1 else 'California'})")
    resp = requests.get(f"{ERDDAP_BASE}/{dataset}.json?{var_str}", timeout=600)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["table"]["rows"], columns=data["table"]["columnNames"])
    for erddap_name, our_name in var_list:
        if erddap_name in df.columns:
            df.rename(columns={erddap_name: our_name}, inplace=True)
    names = [n for _, n in var_list]
    df = df[["latitude", "longitude"] + [n for n in names if n in df.columns]]
    df = df[df[[n for n in names if n in df.columns]].notna().any(axis=1)]
    cache[key] = df.to_dict(orient="list")
    json.dump(cache, open(CACHE_PATH, "w"))
    return df


def _marine_interpolators(times, box, stride):
    """One nearest-neighbour interpolator per marine variable, averaged over times."""
    key = (tuple(times), box, stride)
    if key in _interp:
        return _interp[key]
    frames = []
    for t in times:
        merged = None
        for dataset, var_list in DATASETS.items():
            df = _fetch(dataset, var_list, t, box, stride)
            merged = df if merged is None else merged.merge(
                df, on=["latitude", "longitude"], how="outer")
        frames.append(merged)

    interps = {}
    for var in MARINE_VARS:
        stacks, coords = [], None
        for fr in frames:
            if var not in fr.columns:
                continue
            ok = fr[var].notna().values
            coords = fr.loc[ok, ["latitude", "longitude"]].values
            stacks.append(fr.loc[ok, var].values)
        if stacks:
            interps[var] = NearestNDInterpolator(coords, np.mean(np.column_stack(stacks), axis=1))
    _interp[key] = interps
    return interps


def marine_change(lats, lons):
    """Baseline(2020) -> scenario(avg 2080, 2090) change in the marine variables.

    California points use the native-resolution box; anything outside falls back
    to the 1-degree global grid.
    """
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    inside = ((lats >= CA_BOX[0]) & (lats <= CA_BOX[1]) &
              (lons >= CA_BOX[2]) & (lons <= CA_BOX[3]))

    pres = {v: np.full(len(lats), np.nan) for v in MARINE_VARS}
    scen = {v: np.full(len(lats), np.nan) for v in MARINE_VARS}

    for mask, box, stride in [(inside, CA_BOX, 1),
                              (~inside, (-89.975, 89.975, -179.975, 179.975), GLOBAL_STRIDE)]:
        if not mask.any():
            continue
        pts = np.column_stack([lats[mask], lons[mask]])
        p_int = _marine_interpolators([BASELINE_TIME], box, stride)
        s_int = _marine_interpolators(SCENARIO_TIMES, box, stride)
        for v in MARINE_VARS:
            if v in p_int:
                pres[v][mask] = p_int[v](pts)
            if v in s_int:
                scen[v][mask] = s_int[v](pts)
    return pres, scen


# ═══════════════════════════════════════════════════════════
# The exposure index
# ═══════════════════════════════════════════════════════════

def norm_delta(values, key):
    lo, hi = DELTA_REF[key]
    return np.clip((np.asarray(values, float) - lo) / (hi - lo), 0, 1)


def change_frame(lats, lons, gcm=None, marine=True):
    """Every component change, oriented so that HIGHER always means MORE stress."""
    terr = terrestrial_change(lats, lons, gcm=gcm)
    d = pd.DataFrame(index=np.arange(len(lats)))
    fut1, pres1 = terr[1]
    fut4, pres4 = terr[4]
    fut5, pres5 = terr[5]
    fut14, pres14 = terr[14]
    d["d_bio1"] = fut1 - pres1
    d["d_bio4"] = np.abs(fut4 - pres4)
    d["d_bio5"] = fut5 - pres5
    d["dry_bio14"] = np.clip(-((fut14 - pres14) / (pres14 + 1.0)), 0, None)

    if marine:
        pres, scen = marine_change(lats, lons)
        d["d_sst_mean"] = scen["sst_mean"] - pres["sst_mean"]
        d["d_sst_max"] = scen["sst_max"] - pres["sst_max"]
        d["d_sst_range"] = np.abs(scen["sst_range"] - pres["sst_range"])
        d["d_ph"] = -(scen["ph_mean"] - pres["ph_mean"])
        d["d_o2"] = -(scen["o2_mean"] - pres["o2_mean"])
        d["sst_novelty"] = d["d_sst_mean"] / np.clip(pres["sst_range"], 0.5, None)

    for c in d.columns:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    return d


def compute_climate_exposure(df, gcm=None, return_components=False):
    """Exposure = normalized magnitude of projected climate CHANGE.

    df needs Latitude and Longitude. gcm=None uses the GCM ensemble mean;
    pass a GCM name for a single-model realization.
    """
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    d = change_frame(lats, lons, gcm=gcm, marine=True)

    weights = WEIGHTS if all(k in d.columns for k in MARINE) else WEIGHTS_TERR_ONLY
    comp = pd.DataFrame({k: norm_delta(d[k], k) for k in weights}, index=df.index)
    exposure = sum(comp[k] * w for k, w in weights.items())
    exposure = pd.Series(np.asarray(exposure, float), index=df.index)

    if return_components:
        d.index = df.index
        return exposure, comp, d
    return exposure


def exposure_ensemble(df):
    """Per-GCM exposure -> (mean, sd, min, max, {gcm: values})."""
    per_gcm = {g: compute_climate_exposure(df, gcm=g).values for g in GCMS}
    mat = np.column_stack([per_gcm[g] for g in GCMS])
    return mat.mean(axis=1), mat.std(axis=1), mat.min(axis=1), mat.max(axis=1), per_gcm


if __name__ == "__main__":
    grid = pd.DataFrame({"Latitude": np.linspace(30.5, 42.0, 12)})
    grid["Longitude"] = np.interp(
        grid["Latitude"],
        [30.5, 32.5, 33.0, 34.0, 34.5, 35.0, 36.5, 37.5, 38.5, 40.0, 42.0],
        [-116.1, -117.2, -117.3, -118.5, -120.5, -120.7, -122.0, -122.5, -123.0,
         -124.2, -124.4])
    e, comp, d = compute_climate_exposure(grid, return_components=True)
    out = pd.concat([grid, d.round(3), e.rename("exposure").round(3)], axis=1)
    print(out.to_string(index=False))

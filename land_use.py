#!/usr/bin/env python3
"""
Coastal land-use pressure from NLCD 2021 (30 m) — a non-climate stressor for the
California predictions.

Fetched from the MRLC GeoServer WCS. NLCD covers CONUS only, so the part of the
transect south of the border (~32.5 N, in Baja California) has no data and is
left as NaN rather than silently filled — those points keep a climate-only
vulnerability, and the count is reported.

For each prediction point the land cover within a 5 km radius is
tallied and expressed as fractions OF THE LAND in that window; open water is
excluded, since a coastal window is typically half ocean and would otherwise
dilute every land signal by an arbitrary amount.

  lulc_pressure = (developed + cultivated) / land   0 = wholly natural surroundings
  natural_frac  = (forest+shrub+herbaceous+wetland+barren) / land
  land_frac     = land / window                    how much land is there at all

Rather than 500 point requests, one GeoTIFF strip is pulled per latitude band,
saved under data/LULC_NLCD/, and the per-point windows are cut from it locally.

The transect runs offshore of the NLCD coastline, so a window centred on the
prediction point itself is frequently all water. Each window is therefore snapped
to the nearest land pixel, with the offset recorded as dist_to_land_km.

NLCD classes: 11 water, 12 ice, 21-24 developed, 31 barren, 41-43 forest,
              51-52 shrub, 71-74 herbaceous, 81-82 hay/crops, 90/95 wetlands
"""

import os
import io, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
CACHE = PROC / "land_use_pressure.csv"

# Fetched NLCD strips are kept here so the term is reproducible offline.
def _box_genomics_data():
    """Locate Genomics/data under the Box mount without hardcoding the account folder.

    Box renames the account folder when the display name changes. The previous
    literal path was written before one such rename and silently went stale: the
    callers below guard on .exists(), so instead of failing they fell back to
    coarser data. Globbing survives a rename, and keeps the account folder name
    -- which is an email address -- out of a public repository.
    """
    base = Path.home() / "Library" / "CloudStorage" / "Box-Box"
    for p in sorted(base.glob("*/Genomics/data")):
        if p.is_dir():
            return p
    return base / "Genomics" / "data"


RASTER_DIR = _box_genomics_data() / "LULC_NLCD"
RASTER_DIR.mkdir(parents=True, exist_ok=True)

WCS = "https://www.mrlc.gov/geoserver/mrlc_display/wcs"
COVERAGE = "mrlc_display__NLCD_2021_Land_Cover_L48"
EPSG4326 = "http://www.opengis.net/def/crs/EPSG/0/4326"

WATER = {11, 12}
DEVELOPED = {21, 22, 23, 24}
CULTIVATED = {81, 82}
NATURAL = {31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 90, 95}

RADIUS_KM = 5.0                     # window radius around each prediction point
SEARCH_KM = 25.0                    # how far to look for land before giving up
BAND_DEG = 0.5                      # latitude height of each fetched strip
CONUS_SOUTH = 32.53                 # NLCD stops at the border


def _strip_path(lat_lo, lat_hi, lon_lo, lon_hi):
    return (RASTER_DIR /
            f"nlcd2021_CA_{lat_lo:.1f}-{lat_hi:.1f}N_{abs(lon_hi):.1f}-{abs(lon_lo):.1f}W.tif")


def _fetch_strip(lat_lo, lat_hi, lon_lo, lon_hi):
    """One NLCD GeoTIFF covering a latitude band, saved to disk.

    The longitude extent is part of the cache key: observation records sit at
    different longitudes from the transect, and a strip cached for one extent
    will not cover the other. An existing strip is reused only if it actually
    contains the requested box.
    """
    path = _strip_path(lat_lo, lat_hi, lon_lo, lon_hi)
    if path.exists():
        return rasterio.open(path)
    for cand in RASTER_DIR.glob(f"nlcd2021_CA_{lat_lo:.1f}-{lat_hi:.1f}N_*.tif"):
        src = rasterio.open(cand)
        b = src.bounds
        if (b.left <= lon_lo and b.right >= lon_hi
                and b.bottom <= lat_lo and b.top >= lat_hi):
            return src
        src.close()

    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = t.transform(lon_lo, lat_lo)
    x1, y1 = t.transform(lon_hi, lat_hi)
    params = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage",
              "coverageId": COVERAGE, "format": "image/geotiff",
              "subset": [f"X({x0},{x1})", f"Y({y0},{y1})"],
              "outputCrs": EPSG4326}
    r = requests.get(WCS, params=params, timeout=600)
    if not r.ok or "tif" not in (r.headers.get("content-type") or ""):
        return None
    with rasterio.open(io.BytesIO(r.content)) as src:
        prof = src.profile
        prof.update(driver="GTiff", compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256)
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(src.read())
    return rasterio.open(path)


def _nearest_land(src, land_mask, lat, lon, search_km=SEARCH_KM):
    """Centre of the nearest land pixel, and its distance in km."""
    row, col = src.index(lon, lat)
    row = int(np.clip(row, 0, src.height - 1))
    col = int(np.clip(col, 0, src.width - 1))
    if land_mask[row, col]:
        return lat, lon, 0.0

    px_km = abs(src.res[1]) * 111.0
    rad = int(search_km / max(px_km, 1e-6))
    r0, r1 = max(row - rad, 0), min(row + rad + 1, src.height)
    c0, c1 = max(col - rad, 0), min(col + rad + 1, src.width)
    sub = land_mask[r0:r1, c0:c1]
    if not sub.any():
        return lat, lon, np.nan

    rr, cc = np.nonzero(sub)
    dr = (rr + r0 - row) * px_km
    dc = (cc + c0 - col) * px_km * max(np.cos(np.radians(lat)), 0.1)
    k = int(np.argmin(dr ** 2 + dc ** 2))
    nlon, nlat = src.xy(rr[k] + r0, cc[k] + c0)
    return nlat, nlon, float(np.hypot(dr[k], dc[k]))


def pressure_at(lats, lons, radius_km=RADIUS_KM, use_cache=True):
    """NLCD land-use metrics at each point, cached on rounded coordinates."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    cols = ["lulc_pressure", "natural_frac", "developed_frac", "crop_frac",
            "land_frac", "dist_to_land_km"]
    key = pd.DataFrame({"lat_key": lats.round(4), "lon_key": lons.round(4)})

    cache = (pd.read_csv(CACHE) if (use_cache and CACHE.exists())
             else pd.DataFrame(columns=["lat_key", "lon_key"] + cols))
    merged = key.merge(cache, on=["lat_key", "lon_key"], how="left")
    todo = merged[cols[0]].isna().values & (lats >= CONUS_SOUTH)

    if todo.any():
        idx = np.where(todo)[0]
        print(f"  NLCD 2021: {len(idx)} points, {radius_km:.0f} km radius")
        rows = []
        edges = np.arange(np.floor(lats[idx].min() / BAND_DEG) * BAND_DEG,
                          lats[idx].max() + BAND_DEG, BAND_DEG)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = idx[(lats[idx] >= lo) & (lats[idx] < hi)]
            if len(sel) == 0:
                continue
            # The strip has to be wide enough for the land search, not just for
            # the 5 km window, or the nearest land can fall outside the raster.
            dlat = SEARCH_KM / 111.0
            dlon = SEARCH_KM / (111.0 * np.cos(np.radians(lats[sel].mean())))
            src = _fetch_strip(lo - dlat, hi + dlat,
                               lons[sel].min() - radius_km / 111.0, lons[sel].max() + dlon)
            if src is None:
                print(f"    {lo:.1f}-{hi:.1f} N: no coverage")
                continue
            arr = src.read(1)
            land_mask = (arr > 0) & ~np.isin(arr, list(WATER))
            print(f"    {lo:.1f}-{hi:.1f} N: {arr.shape[1]}x{arr.shape[0]} px, {len(sel)} points",
                  flush=True)
            for i in sel:
                # The transect sits offshore of the NLCD coastline, so a window
                # centred on the point itself is often all water. Snap the window
                # centre to the nearest land pixel and record how far that was.
                clat, clon, dist_km = _nearest_land(src, land_mask, lats[i], lons[i])
                dla = radius_km / 111.0
                dlo = radius_km / (111.0 * max(np.cos(np.radians(clat)), 0.1))
                w = from_bounds(clon - dlo, clat - dla,
                                clon + dlo, clat + dla, src.transform)
                r0 = max(int(w.row_off), 0); c0 = max(int(w.col_off), 0)
                r1 = min(int(w.row_off + w.height), src.height)
                c1 = min(int(w.col_off + w.width), src.width)
                if r1 <= r0 or c1 <= c0:      # window fell outside the strip
                    sub = np.zeros((0, 0), dtype=arr.dtype)
                else:
                    sub = arr[r0:r1, c0:c1]
                v, c = np.unique(sub[sub > 0], return_counts=True)
                counts = dict(zip(v.tolist(), c.tolist()))
                land = sum(n for k, n in counts.items() if k not in WATER)
                allc = sum(counts.values())
                dev = sum(n for k, n in counts.items() if k in DEVELOPED)
                crop = sum(n for k, n in counts.items() if k in CULTIVATED)
                nat = sum(n for k, n in counts.items() if k in NATURAL)
                rows.append({
                    "lat_key": key.lat_key[i], "lon_key": key.lon_key[i],
                    "lulc_pressure": (dev + crop) / land if land else np.nan,
                    "natural_frac": nat / land if land else np.nan,
                    "developed_frac": dev / land if land else np.nan,
                    "crop_frac": crop / land if land else np.nan,
                    "land_frac": land / allc if allc else 0.0,
                    "dist_to_land_km": dist_km,
                })
            src.close()
        if rows:
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
            cache = cache.drop_duplicates(["lat_key", "lon_key"], keep="last")
            cache.to_csv(CACHE, index=False)
            merged = key.merge(cache, on=["lat_key", "lon_key"], how="left")

    out = merged[cols].copy()
    out["outside_nlcd"] = lats < CONUS_SOUTH
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    _p = __import__("04_predict_california")
    lats, lons = _p.make_coastal_grid()
    df = pressure_at(lats, lons)
    df.insert(0, "Longitude", lons)
    df.insert(0, "Latitude", lats)
    n_out = int(df.outside_nlcd.sum())
    print(f"\n{n_out}/{len(df)} points south of {CONUS_SOUTH} N have no NLCD coverage "
          f"(Baja California) — left as NaN")
    ok = df[~df.outside_nlcd]
    print(f"lulc_pressure: mean {ok.lulc_pressure.mean():.3f} "
          f"[{ok.lulc_pressure.min():.3f}, {ok.lulc_pressure.max():.3f}]")
    ok = ok.copy(); ok["bin"] = pd.cut(ok.Latitude, [32, 34, 36, 38, 40, 42])
    print(ok.groupby("bin")[["lulc_pressure", "developed_frac", "crop_frac",
                             "natural_frac", "land_frac"]].mean().round(3).to_string())

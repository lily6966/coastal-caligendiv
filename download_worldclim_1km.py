#!/usr/bin/env python3
"""
Fetch WorldClim v2.1 at 30 arc-seconds (~1 km) for the California study region.

The global 30s files are 10.4 GB (present) and 9.2 GB per CMIP6 GCM — 56 GB for
the set, against 57 GB free on this disk, and the destination is a synced Box
folder. Both servers support HTTP range requests, so instead of downloading whole
globes this reads only the California window over the network and writes local
crops: same 1 km values over the study region, ~520 MB total.

The crops keep WorldClim's grid and CRS exactly, so they are drop-in replacements
for the 10 arc-min rasters, at 18x the resolution.

Output: data/WorldClim_1km/
  present_1970-2000/wc2.1_30s_bio_{b}_CA.tif                     baseline, one band each
  ssp585_2081-2100/wc2.1_30s_bioc_{GCM}_ssp585_..._CA.tif        scenario, 9 bands (BIOS order)
"""

import os, sys, time
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.zip")

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

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


OUT = _box_genomics_data() / "WorldClim_1km"
PRESENT_DIR = OUT / "present_1970-2000"       # historical baseline
FUTURE_DIR = OUT / "ssp585_2081-2100"         # climate change scenario, per GCM
for _d in (PRESENT_DIR, FUTURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CA_BOX = (-126.0, 29.0, -114.0, 43.0)      # west, south, east, north
BIOS = [1, 4, 5, 6, 7, 12, 13, 14, 15]
GCMS = ["BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR", "MIROC6"]
PERIOD = "2081-2100"

PRESENT = ("/vsizip//vsicurl/https://geodata.ucdavis.edu/climate/worldclim/2_1/"
           "base/wc2.1_30s_bio.zip/wc2.1_30s_bio_{b}.tif")
FUTURE = ("/vsicurl/https://geodata.ucdavis.edu/cmip6/30s/{gcm}/ssp585/"
          "wc2.1_30s_bioc_{gcm}_ssp585_" + PERIOD + ".tif")

PROFILE = dict(driver="GTiff", dtype="float32", compress="deflate",
               predictor=2, tiled=True, blockxsize=256, blockysize=256,
               nodata=-3.4e38)


def crop(src, bands, dst_path, label):
    w = from_bounds(*CA_BOX, src.transform)
    t = time.time()
    arr = src.read(bands, window=w).astype("float32")
    arr = np.where(arr < -1e30, np.nan, arr)
    prof = dict(PROFILE, width=arr.shape[-1], height=arr.shape[-2],
                count=arr.shape[0] if arr.ndim == 3 else 1, crs=src.crs,
                transform=src.window_transform(w))
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None])
    mb = dst_path.stat().st_size / 1e6
    print(f"  {label}: {arr.shape[-1]}x{arr.shape[-2]} px, {mb:.1f} MB, {time.time()-t:.0f}s",
          flush=True)


def main():
    print(f"WorldClim 30 arc-sec (~1 km) → {OUT}")
    print(f"California window {CA_BOX}\n")

    print("Present (1970-2000):", flush=True)
    for b in BIOS:
        dst = PRESENT_DIR / f"wc2.1_30s_bio_{b}_CA.tif"
        if dst.exists():
            print(f"  bio{b}: already there", flush=True)
            continue
        with rasterio.open(PRESENT.format(b=b)) as src:
            crop(src, 1, dst, f"bio{b}")

    print(f"\nSSP5-8.5 {PERIOD}, {len(GCMS)} GCMs:", flush=True)
    for gcm in GCMS:
        dst = FUTURE_DIR / f"wc2.1_30s_bioc_{gcm}_ssp585_{PERIOD}_CA.tif"
        if dst.exists():
            print(f"  {gcm}: already there", flush=True)
            continue
        # One windowed read for all needed bands: the source is striped, so
        # separate per-band reads would re-fetch the same byte ranges.
        with rasterio.open(FUTURE.format(gcm=gcm)) as src:
            crop(src, BIOS, dst, gcm)

    for d in (PRESENT_DIR, FUTURE_DIR):
        files = sorted(d.glob("*.tif"))
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"\n{d.name}/: {len(files)} files, {mb:.0f} MB")


if __name__ == "__main__":
    main()

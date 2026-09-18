#!/usr/bin/env python3
"""Offline California land basemap from Natural Earth 10m land polygons.

Replaces online tile providers (which now return "API key required" watermarks)
with a local vector coastline, so map figures are reproducible without network
access or API keys. Coordinates are Web Mercator (EPSG:3857), matching the data.
"""
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

_LAND_PATH = Path(__file__).parent / "data" / "naturalearth" / "ne_10m_land.shp"
_land = None


def _load():
    global _land
    if _land is None:
        _land = gpd.read_file(_LAND_PATH).to_crs(epsg=3857)
    return _land


def add_land(ax, xmin, xmax, ymin, ymax,
             land_color="#e9e9e9", edge_color="#9a9a9a", sea_color="#f4fafe",
             linewidth=0.5):
    """Draw grey land / light sea within the given Web-Mercator extent."""
    ax.set_facecolor(sea_color)
    clip = gpd.clip(_load(), box(xmin, ymin, xmax, ymax))
    if len(clip):
        clip.plot(ax=ax, facecolor=land_color, edgecolor=edge_color,
                  linewidth=linewidth, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

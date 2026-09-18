#!/usr/bin/env python3
"""
Step 9: Rebuild the main pipeline figures on the CHANGE-based (delta-climate)
vulnerability from step 8.

Same layouts and styling as the step 5 / step 6 originals, but every exposure and
vulnerability term comes from the projected CHANGE in climate parameters
(baseline -> SSP5-8.5 end of century) rather than from a single climate state:

  vulnerability_delta = (1 - diversity_norm) * delta_exposure + 0.3 * sigma_norm

Inputs:
  data/processed/vulnerability_scores_delta.csv   (step 8)
  data/processed/california_env.csv               (observed populations, for map markers)

Outputs:
  figures/ecosystem_resilience_delta.{png,pdf}          <- cf. ecosystem_resilience_future
  figures/resilience_map_delta.{png,pdf}                <- cf. resilience_map
  figures/species_vulnerability_ranking_delta.{png,pdf} <- cf. species_vulnerability_ranking
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from pyproj import Transformer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
_f = __import__("06_future_climate")
GCMS = _f.GCMS
PERIOD = _f.PERIOD

VULN = "vulnerability_delta"
VULN_LO = "vulnerability_delta_lo"
VULN_HI = "vulnerability_delta_hi"
EXPO = "delta_exposure"
EXPO_SD = "delta_exposure_sd"
CLASS = "resilience_class_delta"
DIV = "diversity_norm"

CLASS_COLORS = {
    "Resilient": "#2ecc71",
    "At Risk": "#f1c40f",
    "Latent Vulnerability": "#e67e22",
    "Critical": "#e74c3c",
}
VULN_CMAP = LinearSegmentedColormap.from_list(
    "vuln", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])

CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
_tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
CA_XMIN, CA_YMIN = _tr.transform(CA_LON_MIN, CA_LAT_MIN)
CA_XMAX, CA_YMAX = _tr.transform(CA_LON_MAX, CA_LAT_MAX)

SUBTITLE = (f"Exposure = projected CHANGE in climate parameters "
            f"(baseline → SSP5-8.5 {PERIOD})\n"
            f"Terrestrial: {len(GCMS)}-GCM ensemble | "
            f"Marine: Bio-ORACLE SSP5-8.5, 2020 → avg(2080, 2090)")


def _setup_ca_panel(ax, alpha=0.6):
    ax.set_xlim(CA_XMIN, CA_XMAX)
    ax.set_ylim(CA_YMIN, CA_YMAX)
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7, alpha=alpha)
    except Exception:
        ax.set_facecolor("#f0f0f0")
    ax.set_xlim(CA_XMIN, CA_XMAX)
    ax.set_ylim(CA_YMIN, CA_YMAX)


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: figures/{name}.png + .pdf")


# ═══════════════════════════════════════════════════════════
# Figure 1: ecosystem-level resilience (cf. ecosystem_resilience_future)
# ═══════════════════════════════════════════════════════════

def fig_ecosystem(res):
    lat_edges = np.linspace(res["Latitude"].min(), res["Latitude"].max(), 76)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    # coastline for placing the bin markers
    coast_lats = np.sort(res["Latitude"].unique())
    coast_lons = np.array([res.loc[res["Latitude"] == la, "Longitude"].iloc[0]
                           for la in coast_lats])

    rows = []
    for i in range(len(lat_edges) - 1):
        pts = res[(res["Latitude"] >= lat_edges[i]) & (res["Latitude"] < lat_edges[i + 1])]
        if len(pts) == 0:
            continue
        rows.append({
            "lat_center": lat_centers[i],
            "lon_center": np.interp(lat_centers[i], coast_lats, coast_lons),
            "n_species": pts["species"].nunique(),
            "mean_vulnerability": pts[VULN].mean(),
            "mean_vulnerability_lo": pts[VULN_LO].mean(),
            "mean_vulnerability_hi": pts[VULN_HI].mean(),
            "mean_diversity": pts[DIV].mean(),
            "mean_exposure": pts[EXPO].mean(),
            "mean_exposure_std": pts[EXPO_SD].mean(),
            "mean_uncertainty": pts["sigma_norm"].mean(),
            **{f"pct_{k}": (pts[CLASS] == lbl).mean() for k, lbl in
               [("critical", "Critical"), ("latent", "Latent Vulnerability"),
                ("at_risk", "At Risk"), ("resilient", "Resilient")]},
        })
    eco = pd.DataFrame(rows)

    fig = plt.figure(figsize=(28, 24))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.25, wspace=0.2)

    # ── Panel A: map ──
    ax_map = fig.add_subplot(gs[:, 0])
    _setup_ca_panel(ax_map, alpha=0.5)
    eco_gdf = gpd.GeoDataFrame(
        eco, geometry=gpd.points_from_xy(eco["lon_center"], eco["lat_center"]),
        crs="EPSG:4326").to_crs(epsg=3857)
    sizes = eco["n_species"] / eco["n_species"].max() * 200 + 50
    sc = ax_map.scatter(eco_gdf.geometry.x, eco_gdf.geometry.y,
                        c=eco["mean_vulnerability"], cmap=VULN_CMAP, s=sizes,
                        alpha=0.85, edgecolors="black", linewidth=0.4,
                        vmin=0.1, vmax=0.7, zorder=3)
    cb = plt.colorbar(sc, ax=ax_map, shrink=0.5, pad=0.02)
    cb.set_label("Mean vulnerability to climate CHANGE (ensemble)", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax_map.set_title("Ecosystem Resilience to Projected Climate Change\n"
                     f"({len(GCMS)}-GCM Ensemble Mean)", fontsize=16, fontweight="bold")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for ns, label in [(5, "5 spp"), (15, "15 spp"), (25, "25 spp")]:
        ax_map.scatter([], [], s=ns / eco["n_species"].max() * 200 + 50, c="gray",
                       alpha=0.6, edgecolors="black", linewidth=0.4, label=label)
    ax_map.legend(title="Species coverage", loc="lower left", fontsize=11,
                  title_fontsize=12, frameon=True, fancybox=True)

    # ── Panel B: class composition ──
    ax_stack = fig.add_subplot(gs[0, 1])
    c, lat = eco, eco["lat_center"]
    ax_stack.fill_between(lat, 0, c["pct_critical"], color="#e74c3c", alpha=0.8,
                          label="Critical")
    ax_stack.fill_between(lat, c["pct_critical"], c["pct_critical"] + c["pct_latent"],
                          color="#e67e22", alpha=0.8, label="Latent Vulnerability")
    ax_stack.fill_between(lat, c["pct_critical"] + c["pct_latent"],
                          c["pct_critical"] + c["pct_latent"] + c["pct_at_risk"],
                          color="#f1c40f", alpha=0.8, label="At Risk")
    ax_stack.fill_between(lat, c["pct_critical"] + c["pct_latent"] + c["pct_at_risk"],
                          1.0, color="#2ecc71", alpha=0.8, label="Resilient")
    ax_count = ax_stack.twinx()
    ax_count.plot(lat, c["n_species"], "k-", linewidth=2, alpha=0.6)
    ax_count.set_ylabel("Number of species", fontsize=12)
    ax_stack.set_xlabel("Latitude", fontsize=12)
    ax_stack.set_ylabel("Proportion of species", fontsize=12)
    ax_stack.set_title("Resilience Class Composition\n(change-based exposure)",
                       fontsize=14, fontweight="bold")
    ax_stack.set_ylim([0, 1])
    ax_stack.legend(loc="upper right", fontsize=10, frameon=True)
    ax_stack.grid(True, alpha=0.2, axis="x")

    # ── Panel C: components ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(lat, 1 - c["mean_diversity"], "b-o", markersize=3, linewidth=2,
              alpha=0.7, label="Diversity deficit (1 - div_norm)")
    ax_c.plot(lat, c["mean_exposure"], "r-s", markersize=3, linewidth=2, alpha=0.7,
              label="Change-based exposure (ensemble mean)")
    ax_c.fill_between(lat, c["mean_exposure"] - c["mean_exposure_std"],
                      c["mean_exposure"] + c["mean_exposure_std"],
                      color="red", alpha=0.15, label="Inter-model ±1 SD")
    ax_c.plot(lat, c["mean_uncertainty"], color="gray", linestyle="--", linewidth=1.5,
              alpha=0.6, label="Model uncertainty")
    ax_c.plot(lat, c["mean_vulnerability"], "k-", linewidth=2.5, alpha=0.9,
              label="Vulnerability (ensemble mean)")
    ax_c.fill_between(lat, c["mean_vulnerability_lo"], c["mean_vulnerability_hi"],
                      color="black", alpha=0.1, label="Vulnerability (GCM range)")
    ax_c.set_xlabel("Latitude", fontsize=12)
    ax_c.set_ylabel("Score (0-1)", fontsize=12)
    ax_c.set_title("Vulnerability Components\n(with inter-model uncertainty)",
                   fontsize=14, fontweight="bold")
    ax_c.legend(fontsize=9, loc="best", frameon=True)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim([0, 1])

    fig.suptitle(
        "Ecosystem-Level Vulnerability to Climate CHANGE — California Coast\n"
        + SUBTITLE
        + f"\n{res['species'].nunique()} species, {len(res)} predictions",
        fontsize=17, fontweight="bold", y=0.98)
    _save(fig, "ecosystem_resilience_delta")
    return eco


# ═══════════════════════════════════════════════════════════
# Figure 2: per-species resilience maps (cf. resilience_map)
# ═══════════════════════════════════════════════════════════

def fig_species_maps(res, cali_df):
    top6 = (res.groupby("species")
              .agg(n_obs=("n_obs", "first"), metric_type=("metric_type", "first"))
              .reset_index().nlargest(6, "n_obs"))

    fig = plt.figure(figsize=(24, 32))
    gs = GridSpec(3, 3, figure=fig, hspace=0.08, wspace=0.05,
                  top=0.93, bottom=0.06, left=0.02, right=0.98)

    for idx, (_, row) in enumerate(top6.iterrows()):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sp = row["species"]
        sp_data = res[res["species"] == sp]
        _setup_ca_panel(ax)

        gdf = gpd.GeoDataFrame(
            sp_data, geometry=gpd.points_from_xy(sp_data["Longitude"], sp_data["Latitude"]),
            crs="EPSG:4326").to_crs(epsg=3857)
        for cls, color in CLASS_COLORS.items():
            m = gdf[CLASS] == cls
            if m.sum():
                gdf[m].plot(ax=ax, color=color, markersize=60, alpha=0.75,
                            edgecolors="black", linewidth=0.3, zorder=3)

        obs = cali_df[cali_df["species"] == sp]
        if len(obs):
            gpd.GeoDataFrame(
                obs, geometry=gpd.points_from_xy(obs["Longitude"], obs["Latitude"]),
                crs="EPSG:4326").to_crs(epsg=3857).plot(
                    ax=ax, color="black", marker="^", markersize=90, zorder=4,
                    edgecolors="white", linewidth=0.8)

        ax.set_xlim(CA_XMIN, CA_XMAX)
        ax.set_ylim(CA_YMIN, CA_YMAX)
        ax.set_title(f"{sp}\n({row['metric_type']}, n={int(row['n_obs'])})",
                     fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([]); ax.set_yticks([])

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=22,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Resilient\n(High diversity, Low change)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f", markersize=22,
               markeredgecolor="black", markeredgewidth=0.5,
               label="At Risk\n(High diversity, High change)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=22,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Latent Vulnerability\n(Low diversity, Low change)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=22,
               markeredgecolor="black", markeredgewidth=0.5,
               label="Critical\n(Low diversity, High change)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=20,
               markeredgecolor="white", markeredgewidth=0.8, label="Observed population"),
    ]
    ax_leg.legend(handles=handles, loc="center", ncol=5, fontsize=14, frameon=True,
                  fancybox=True, shadow=True, handletextpad=0.5, columnspacing=2.0)

    fig.suptitle("Resilience Classification Under Projected Climate CHANGE\n" + SUBTITLE,
                 fontsize=20, fontweight="bold")
    _save(fig, "resilience_map_delta")


# ═══════════════════════════════════════════════════════════
# Figure 3: species vulnerability ranking (cf. species_vulnerability_ranking)
# ═══════════════════════════════════════════════════════════

def fig_ranking(res):
    sp = res.groupby("species").agg(
        mean_vulnerability=(VULN, "mean"),
        mean_vulnerability_lo=(VULN_LO, "mean"),
        mean_vulnerability_hi=(VULN_HI, "mean"),
        mean_exposure=(EXPO, "mean"),
        mean_diversity=(DIV, "mean"),
        mean_uncertainty=("sigma_norm", "mean"),
        pct_critical=(CLASS, lambda x: (x == "Critical").mean()),
        metric_type=("metric_type", "first"),
        n_obs=("n_obs", "first"),
    ).reset_index().sort_values("mean_vulnerability", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#e74c3c" if r["pct_critical"] > 0.5 else
              "#e67e22" if r["pct_critical"] > 0.3 else
              "#f1c40f" if r["pct_critical"] > 0.1 else
              "#2ecc71" for _, r in sp.iterrows()]
    ax.barh(range(len(sp)), sp["mean_vulnerability"], color=colors, alpha=0.8)
    ax.errorbar(sp["mean_vulnerability"], range(len(sp)),
                xerr=[sp["mean_vulnerability"] - sp["mean_vulnerability_lo"],
                      sp["mean_vulnerability_hi"] - sp["mean_vulnerability"]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([f"{r['species']} ({r['metric_type']}, n={r['n_obs']:.0f})"
                        for _, r in sp.iterrows()], fontsize=8)
    ax.set_xlabel("Mean vulnerability = (1 - diversity) × Δ-climate exposure + 0.3σ")
    ax.set_title("Species Vulnerability to Projected Climate CHANGE\n"
                 "(Green=Resilient, Yellow=At Risk, Orange=Vulnerable, Red=>50% Critical; "
                 "whiskers = across-GCM range)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save(fig, "species_vulnerability_ranking_delta")
    return sp


def main():
    res = pd.read_csv(PROC / "vulnerability_scores_delta.csv")
    cali_df = pd.read_csv(PROC / "california_env.csv")
    print(f"Loaded {len(res)} predictions, {res['species'].nunique()} species")
    print(f"  mean {VULN} = {res[VULN].mean():.4f} "
          f"[{res[VULN].min():.3f}, {res[VULN].max():.3f}]")

    eco = fig_ecosystem(res)
    fig_species_maps(res, cali_df)
    sp = fig_ranking(res)

    print(f"\n  latitude-bin vulnerability: "
          f"{eco['mean_vulnerability'].min():.3f} (south) → "
          f"{eco['mean_vulnerability'].max():.3f} (north)")
    print(f"  most vulnerable species: {sp.iloc[-1]['species']} "
          f"({sp.iloc[-1]['mean_vulnerability']:.3f})")
    print(f"  least vulnerable species: {sp.iloc[0]['species']} "
          f"({sp.iloc[0]['mean_vulnerability']:.3f})")


if __name__ == "__main__":
    main()

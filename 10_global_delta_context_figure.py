#!/usr/bin/env python3
"""
Step 10: Global-context figure — California He species inside the global cloud of
19,163 populations, on the pipeline's exposure axis.

The original placed California's He species inside the global cloud of 19,163
populations using the *state* exposure index, then moved them right in panel (d)
as the future climate got hotter. Here the x-axis is the projected CHANGE in
climate parameters, computed the same way for the global populations as for the
California coast, so the global cloud is a like-for-like backdrop.

Because exposure is now a single fixed quantity per site (how much that site
changes), the current -> future arrow in panel (d) is vertical: what moves is the
CNP-projected genetic diversity, not the exposure.

Global delta inputs:
  terrestrial  bio1, bio4, bio5, bio14 present (global_train_env.csv)
               -> 5-GCM CMIP6 ensemble 2081-2100 (local WorldClim rasters)
  marine       Bio-ORACLE SSP5-8.5, 2020 -> avg(2080, 2090), fetched globally on a
               1-degree stride and matched to each population by nearest neighbour

Both sides use the same definition of exposure, climate_delta.compute_climate_exposure
— the projected change between the historical baseline and SSP5-8.5 end of century.
California's values are read straight from vulnerability_scores.csv; the global
populations are scored by the same function.

Diversity is plotted RAW on both sides — observed He for the global populations,
predicted He for California — with no normalization at all. He is already bounded
0-1, so the two are directly comparable without rescaling. The quadrant divider is
the raw He that the old global reference mapped to 0.5, so quadrant membership is
unchanged. This affects only this figure's axis: vulnerability_scores.csv still
scores diversity within species.

Panels:
  a) global populations, density in (delta-exposure, He) space
  b) global cloud + California He species, present-day diversity
  c) global cloud + California He species, CNP-projected 2100 diversity
  d) per-species arrows: present -> projected 2100 diversity at fixed delta-exposure

Outputs:
  figures/div_vs_exposure_He_combined.{pdf,png}
  data/processed/global_delta_exposure.csv
"""

import json, sys, warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RASTER_DIR = ROOT / "data" / "env_rasters"

sys.path.insert(0, str(ROOT))
import climate_delta
GCMS = climate_delta.GCMS
PERIOD = climate_delta.PERIOD

SPECIES_COLORS = [plt.cm.tab10(i) for i in [0, 1, 2, 4, 5, 7, 8, 9]]


def build_global_exposure():
    """Score the global populations with the pipeline's exposure definition."""
    g = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    print(f"  global populations: {len(g)}")
    g["climate_exposure"] = climate_delta.compute_climate_exposure(g).values

    # Diversity is plotted RAW — observed He for the global populations, predicted
    # He for California — with no normalization on either side. He is already a
    # 0-1 quantity, so the axis needs no rescaling to be comparable.
    #
    # The quadrant divider is drawn at the raw He that used to map to 0.5 under
    # the old global reference, i.e. the midpoint of its 2nd-98th percentile range,
    # so quadrant membership is unchanged by dropping the normalization.
    HE_LO, HE_HI = g["gen_div"].quantile(0.02), g["gen_div"].quantile(0.98)
    divider = (HE_LO + HE_HI) / 2
    g["diversity_raw"] = g["gen_div"]
    print(f"  raw global He: median {g['gen_div'].median():.3f}, "
          f"quadrant divider at He = {divider:.3f}")
    return g, divider

    g[["species", "Latitude", "Longitude", "gen_div",
       "climate_exposure"]].to_csv(PROC / "global_exposure.csv", index=False)
    print(f"  saved: {PROC / 'global_exposure.csv'}")
    print(f"  global exposure: mean={g['climate_exposure'].mean():.3f} "
          f"[{g['climate_exposure'].min():.3f}, {g['climate_exposure'].max():.3f}]")


# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("GLOBAL CONTEXT FIGURE ON CHANGE-BASED EXPOSURE")
    print("=" * 64)

    g, HE_DIVIDER = build_global_exposure()

    # California keeps its raw predicted He too — pred_mu now, pred_mu_future in 2100.
    ca = pd.read_csv(PROC / "vulnerability_scores.csv")
    fd = pd.read_csv(PROC / "future_diversity.csv")
    ca["diversity_raw"] = ca["pred_mu"]
    ca["diversity_raw_future"] = fd["pred_mu_future"].values

    he = ca[ca["metric_type"] == "He"].copy()
    species = sorted(he["species"].unique())
    print(f"  He species in figure: {len(species)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    gx, gy = g["climate_exposure"].values, g["diversity_raw"].values

    def quadrant_frame(ax, xlabel):
        ax.axhline(HE_DIVIDER, color="gray", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(0.5, color="gray", ls="--", lw=1.2, alpha=0.8)
        hi_y, lo_y = HE_DIVIDER + 0.30, HE_DIVIDER - 0.30
        ax.text(0.18, hi_y, "Resilient", color="#2ecc71", fontsize=12,
                fontweight="bold", alpha=0.9)
        ax.text(0.72, hi_y, "At Risk", color="#f1c40f", fontsize=12,
                fontweight="bold", alpha=0.9)
        ax.text(0.13, lo_y, "Latent\nVulnerable", color="#e67e22", fontsize=12,
                fontweight="bold", alpha=0.9)
        ax.text(0.72, lo_y, "Critical", color="#e74c3c", fontsize=12,
                fontweight="bold", alpha=0.9)
        ax.text(0.995, HE_DIVIDER, f" He = {HE_DIVIDER:.2f}", fontsize=7,
                color="gray", va="bottom", ha="right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(alpha=0.25)

    XLAB = "Climate exposure = projected change, SSP5-8.5 " + PERIOD

    # ── a) global density ──
    ax = axes[0, 0]
    hb = ax.hexbin(gx, gy, gridsize=45, cmap="YlOrRd", mincnt=1, extent=(0, 1, 0, 1))
    cb = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Population count", fontsize=10)
    quadrant_frame(ax, XLAB)
    ax.set_ylabel("He (heterozygosity, unnormalized)", fontsize=11)
    ax.text(0.95, 0.95, "a)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="right", va="top")

    # ── b, c) global backdrop + California ──
    for ax, ycol, letter, title in [
        (axes[0, 1], "diversity_raw", "b)", "present-day diversity"),
        (axes[1, 0], "diversity_raw_future", "c)", "projected 2100 diversity"),
    ]:
        ax.hexbin(gx, gy, gridsize=45, cmap="Blues", mincnt=1, alpha=0.45,
                  extent=(0, 1, 0, 1))
        for sp, color in zip(species, SPECIES_COLORS):
            s = he[he["species"] == sp]
            ax.scatter(s["climate_exposure"], s[ycol], s=42, color=color,
                       alpha=0.85, edgecolors="black", linewidth=0.4, zorder=3)
        quadrant_frame(ax, XLAB)
        ax.set_ylabel("He (heterozygosity, unnormalized)", fontsize=11)
        ax.set_title(f"California He species — {title}", fontsize=11)
        ax.text(0.95, 0.95, letter, transform=ax.transAxes, fontsize=15,
                fontweight="bold", ha="right", va="top")

    # ── d) species-level shift in diversity at fixed exposure ──
    ax = axes[1, 1]
    for i, (sp, color) in enumerate(zip(species, SPECIES_COLORS)):
        s = he[he["species"] == sp]
        x = s["climate_exposure"].mean()
        y0, y1 = s["diversity_raw"].mean(), s["diversity_raw_future"].mean()
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, alpha=0.85))
        ax.scatter([x], [y0], s=110, color=color, marker="o", edgecolors="black",
                   linewidth=0.5, zorder=3)
        ax.scatter([x], [y1], s=130, color=color, marker="D", edgecolors="black",
                   linewidth=0.5, zorder=3)
        # stagger the labels — several species land on nearly the same point
        genus, epithet = sp.split()[0], sp.split()[-1]
        dx, dy = (0.015, -0.022) if i % 2 else (0.015, 0.022)
        ax.annotate(f"{genus[0]}. {epithet}", xy=(x, y1), xytext=(x + dx, y1 + dy),
                    fontsize=8, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.6))
    quadrant_frame(ax, "Δ-climate exposure")
    ax.set_ylabel("He (heterozygosity, unnormalized)", fontsize=11)
    ax.set_title("Projected diversity shift at fixed Δ-exposure", fontsize=11)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=11,
               markeredgecolor="black", label="Present diversity"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markersize=11,
               markeredgecolor="black", label="Projected 2100 (CNP)")],
        loc="upper left", fontsize=9, frameon=True)
    ax.text(0.95, 0.95, "d)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="right", va="top")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=11,
                      markeredgecolor="black", markeredgewidth=0.4, label=sp)
               for sp, c in zip(species, SPECIES_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Genetic Diversity vs Projected Climate CHANGE — California He species "
        "in global context\n"
        f"Exposure = normalized change in 10 climate parameters, baseline → SSP5-8.5 "
        f"{PERIOD} ({len(GCMS)}-GCM terrestrial ensemble + Bio-ORACLE marine); "
        f"global cloud = {len(g):,} populations",
        fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(FIG_DIR / "div_vs_exposure_He_combined.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "div_vs_exposure_He_combined.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    print("\n  saved: figures/div_vs_exposure_He_combined.pdf + .png")

    print("\n  California He species (exposure, raw He now → 2100):")
    for sp in species:
        s = he[he["species"] == sp]
        print(f"    {sp[:28]:30s} exp={s['climate_exposure'].mean():.3f}  "
              f"He {s['diversity_raw'].mean():.3f} → {s['diversity_raw_future'].mean():.3f} "
              f"({s['diversity_raw_future'].mean() - s['diversity_raw'].mean():+.3f})")
    print(f"\n  global mean exposure {g['climate_exposure'].mean():.3f} vs "
          f"California He species {he['climate_exposure'].mean():.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Supplementary figure: per-GCM exposure comparison.

Reads vulnerability_scores_future.csv (produced by 06_future_climate.py)
and generates a multi-panel figure showing individual GCM contributions.

Panels:
  A — Per-GCM exposure along latitude (individual lines) + historical baseline
  B — Mean exposure per GCM (bar chart) with spatial SD error bars
  C — Current→Future shift per species (arrows), ensemble mean + GCM range
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG_DIR = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
GCMS = ["BCC-CSM2-MR", "CanESM5", "CNRM-CM6-1", "IPSL-CM6A-LR", "MIROC6"]
GCM_COLORS = {
    "BCC-CSM2-MR":  "#1b9e77",
    "CanESM5":       "#d95f02",
    "CNRM-CM6-1":   "#7570b3",
    "IPSL-CM6A-LR": "#e7298a",
    "MIROC6":        "#66a61e",
}
GCM_SHORT = {
    "BCC-CSM2-MR":  "BCC",
    "CanESM5":       "CanESM5",
    "CNRM-CM6-1":   "CNRM",
    "IPSL-CM6A-LR": "IPSL",
    "MIROC6":        "MIROC6",
}


def main():
    df = pd.read_csv(PROC / "vulnerability_scores_future.csv")
    print(f"Loaded {len(df)} records, {df['species'].nunique()} species")

    # Check required columns
    gcm_cols = [f"exposure_{g}" for g in GCMS]
    missing = [c for c in gcm_cols if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        return

    # ── Latitude binning for Panel A ──
    lat_bins = np.linspace(df["Latitude"].min(), df["Latitude"].max(), 50)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    lat_stats = []
    for i in range(len(lat_bins) - 1):
        mask = (df["Latitude"] >= lat_bins[i]) & (df["Latitude"] < lat_bins[i + 1])
        pts = df[mask]
        if len(pts) == 0:
            continue
        row = {"lat": lat_centers[i]}
        row["ensemble"] = pts["climate_exposure_future"].mean()
        for gcm in GCMS:
            row[gcm] = pts[f"exposure_{gcm}"].mean()
        lat_stats.append(row)

    lat_df = pd.DataFrame(lat_stats)

    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # ── Panel A: Per-GCM exposure along latitude ──
    ax = axes[0]
    for gcm in GCMS:
        ax.plot(lat_df["lat"], lat_df[gcm],
                color=GCM_COLORS[gcm], linewidth=1.5, alpha=0.8,
                label=GCM_SHORT[gcm])
    ax.plot(lat_df["lat"], lat_df["ensemble"],
            "k-", linewidth=3, alpha=0.9, label="Ensemble mean", zorder=4)

    # Shade the GCM range
    gcm_arr = lat_df[GCMS].values
    ax.fill_between(lat_df["lat"], gcm_arr.min(axis=1), gcm_arr.max(axis=1),
                    color="gray", alpha=0.15, label="GCM range")

    ax.set_xlabel("Latitude (°N)", fontsize=12)
    ax.set_ylabel("Climate exposure (Δ baseline → SSP5-8.5)", fontsize=12)
    ax.set_title("(A) Change-Based Exposure Along Coast\nPer-GCM Lines + Ensemble",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.2, 0.9])

    # ── Panel B: Bar chart — mean exposure per GCM ──
    ax = axes[1]
    gcm_means = [df[f"exposure_{g}"].mean() for g in GCMS]
    gcm_sds = [df[f"exposure_{g}"].std() for g in GCMS]
    x_pos = np.arange(len(GCMS) + 1)
    labels = [GCM_SHORT[g] for g in GCMS] + ["Ensemble"]
    means = gcm_means + [df["climate_exposure_future"].mean()]
    sds = gcm_sds + [df["climate_exposure_future"].std()]
    colors = [GCM_COLORS[g] for g in GCMS] + ["black"]

    bars = ax.bar(x_pos, means, yerr=sds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean change-based exposure", fontsize=12)
    ax.set_title("(B) Mean Δ-Climate Exposure by GCM\n(error bars = spatial SD across populations)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim([0, 1.0])

    # Add value labels
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Horizontal line at 0.5 threshold
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.text(len(x_pos) - 0.5, 0.51, "high exposure\nthreshold", fontsize=8,
            color="gray", ha="right", va="bottom")

    # ── Panel C: Species-level current→future shift ──
    ax = axes[2]

    he_df = df[df["metric_type"] == "He"]
    species_list = sorted(he_df["species"].unique())
    sp_colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))

    for i, sp in enumerate(species_list):
        sp_d = he_df[he_df["species"] == sp]
        exp_mean = sp_d["climate_exposure_future"].mean()
        exp_min = sp_d["climate_exposure_future_min"].mean()
        exp_max = sp_d["climate_exposure_future_max"].mean()
        div_now = sp_d["diversity_norm"].mean()
        div_fut = sp_d["diversity_norm_future"].mean()

        # Exposure is one fixed quantity per site (the projected change), so what
        # moves between now and 2100 is diversity: the arrow is vertical.
        ax.annotate("", xy=(exp_mean, div_fut), xytext=(exp_mean, div_now),
                    arrowprops=dict(arrowstyle="->", color=sp_colors[i],
                                    lw=1.8, alpha=0.7))

        # GCM range in exposure, drawn at the arrow tip
        ax.plot([exp_min, exp_max], [div_fut, div_fut],
                color=sp_colors[i], linewidth=4, alpha=0.3, solid_capstyle="round")

        ax.scatter(exp_mean, div_now, s=60, color=sp_colors[i],
                   edgecolors="black", linewidth=0.5, zorder=5, marker="o")
        ax.scatter(exp_mean, div_fut, s=60, color=sp_colors[i],
                   edgecolors="black", linewidth=0.5, zorder=5, marker="D")

        ax.text(exp_max + 0.01, div_fut, sp.split()[-1][:8],
                fontsize=7, va="center", color=sp_colors[i], alpha=0.8)

    ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    ax.axvline(0.5, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Climate exposure (Δ baseline → SSP5-8.5)", fontsize=12)
    ax.set_ylabel("He (normalized within species)", fontsize=12)
    ax.set_title("(C) Species Diversity Shift at Fixed Exposure\n"
                 "(arrows: present → 2100 diversity, bars: GCM range in exposure)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
               markersize=8, linestyle="None", label="Present diversity"),
        Line2D([0], [0], marker="D", color="gray", markerfacecolor="gray",
               markersize=8, linestyle="None", label="Projected 2100 diversity"),
        Line2D([0], [0], color="gray", linewidth=4, alpha=0.3, label="GCM range"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left", frameon=True)

    for txt, x, y, c in [("Resilient", 0.25, 0.75, "#2ecc71"),
                           ("At Risk", 0.75, 0.75, "#f1c40f"),
                           ("Latent\nVulnerable", 0.25, 0.25, "#e67e22"),
                           ("Critical", 0.75, 0.25, "#e74c3c")]:
        ax.text(x, y, txt, ha="center", fontsize=10, color=c, alpha=0.3, fontweight="bold")

    plt.suptitle(
        "Supplementary: Inter-Model Comparison of Change-Based Climate Exposure\n"
        "5 CMIP6 GCMs | SSP5-8.5 2081-2100 | Bio-ORACLE avg(2080, 2090)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"gcm_exposure_comparison.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'gcm_exposure_comparison.pdf'}")

    # Print summary table
    print("\n" + "=" * 60)
    print("GCM EXPOSURE SUMMARY")
    print("=" * 60)
    print(f"{'GCM':20s} {'Mean':>8s} {'SD':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 60)
    for gcm in GCMS:
        col = f"exposure_{gcm}"
        print(f"{GCM_SHORT[gcm]:20s} {df[col].mean():8.4f} {df[col].std():8.4f} "
              f"{df[col].min():8.4f} {df[col].max():8.4f}")
    ens_mean = df["climate_exposure_future"].mean()
    ens_sd = df["climate_exposure_future"].std()
    print(f"{'Ensemble':20s} {ens_mean:8.4f} {ens_sd:8.4f} "
          f"{df['climate_exposure_future'].min():8.4f} {df['climate_exposure_future'].max():8.4f}")

    inter_sd = df["climate_exposure_future_std"].mean()
    inter_range = (df["climate_exposure_future_max"] - df["climate_exposure_future_min"]).mean()
    print(f"\nInter-model SD (mean across populations): {inter_sd:.4f}")
    print(f"Inter-model range (mean across populations): {inter_range:.4f}")
    spread = (df[[f"exposure_{g}" for g in GCMS]].mean().max() -
              df[[f"exposure_{g}" for g in GCMS]].mean().min())
    print(f"Spread between the warmest and coolest GCM: {spread:.4f}")


if __name__ == "__main__":
    main()

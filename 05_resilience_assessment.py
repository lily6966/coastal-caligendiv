#!/usr/bin/env python3
"""
Step 5: Climate change resilience assessment.

Combines CNP-predicted genetic diversity with climate exposure to identify
vulnerable populations and species.

Exposure is the projected CHANGE in climate parameters between the historical
baseline and SSP5-8.5 end of century (see climate_delta.py), computed per GCM and
ensembled — not the absolute value of those parameters in a single time slice.
The superseded state-based indices are still recorded as climate_exposure_v1 and
climate_exposure_state for the comparison figures.

Vulnerability combines that climate change signal with coastal land-use pressure
from NLCD 2021 (30 m), weighted LULC_WEIGHT; land use is not climate, so it stays
its own term. Points outside NLCD coverage (south of the border) fall back to a
climate-only stressor and are flagged by lulc_available.

Predictions are raw He / pi throughout. The vulnerability score normalizes
diversity against a CROSS-SPECIES reference (global 2nd-98th percentile He, CA
2nd-98th percentile pi). Within-species min-max is retained as
diversity_norm_within for plotting only — see the note in main().

Outputs:
  data/processed/vulnerability_scores.csv
  figures/resilience_map.png
  figures/species_vulnerability_ranking.png
  figures/diversity_vs_exposure.png
  figures/global_vs_california.png
"""

import pickle, sys, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
MODEL_DIR = ROOT / "models"
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG_DIR = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
ConditionalNeuralProcess = _m.ConditionalNeuralProcess
FeatureProcessor = _m.FeatureProcessor
NUMERIC_FEATURES = _m.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _m.CATEGORICAL_FEATURES
REPR_DIM = _m.REPR_DIM

_p = __import__("04_predict_california")
import climate_delta
import land_use
load_model_and_processor = _p.load_model_and_processor
prepare_context = _p.prepare_context
predict_at_locations = _p.predict_at_locations
make_target_df = _p.make_target_df
make_coastal_grid = _p.make_coastal_grid

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_climate_exposure_v1(df):
    """ORIGINAL exposure index (v1) — kept for comparison.

    Simple weighted average of 4 components, locally normalized.
    """
    exposure = pd.DataFrame(index=df.index)

    def norm01(series):
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-8:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    exposure["temp_stress"] = norm01(df["bio5"]) if "bio5" in df.columns else 0.5
    exposure["seasonality_stress"] = norm01(df["bio4"]) if "bio4" in df.columns else 0.5
    exposure["precip_stress"] = 1 - norm01(df["bio14"]) if "bio14" in df.columns else 0.5
    if "sst_max" in df.columns and df["sst_max"].notna().any():
        exposure["marine_thermal_stress"] = norm01(df["sst_max"].fillna(df["sst_max"].median()))
    else:
        exposure["marine_thermal_stress"] = 0.5

    weights = {"temp_stress": 0.3, "seasonality_stress": 0.2,
               "precip_stress": 0.2, "marine_thermal_stress": 0.3}
    return sum(exposure[col] * w for col, w in weights.items())


# ── Global reference ranges for absolute normalization ──
# From WorldClim + Bio-ORACLE across global + California datasets
GLOBAL_REF = {
    "bio5":  (-7.5, 43.1),    # Max temp warmest month (°C)
    "bio4":  (0.0, 1920.0),   # Temperature seasonality (sd × 100)
    "bio14": (0.0, 325.0),    # Precip driest month (mm)
    "bio7":  (4.0, 51.0),     # Temp annual range (°C)
    "bio1":  (-20.0, 30.0),   # Mean annual temp (°C)
    "sst_max": (14.5, 23.5),  # Max SST California range (°C)
    "sst_range": (5.5, 11.2), # SST annual range California (°C)
    "ph_mean": (7.91, 8.04),  # Ocean pH
    "o2_mean": (248.0, 267.2),# Dissolved O₂ (mol/m³)
}


def norm01_abs(series, col_name):
    """Normalize using global reference ranges for absolute comparability."""
    lo, hi = GLOBAL_REF.get(col_name, (series.min(), series.max()))
    if hi - lo < 1e-8:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def compute_climate_exposure_state(df):
    """SUPERSEDED state-based index (v2) — kept only for comparison figures.

    Scores the absolute value of climate variables in a single time slice, so a
    site ranks high because it is already warm. Exposure is now defined as the
    projected CHANGE in those parameters instead; see climate_delta.py.

    Separates marine and terrestrial stressors.
    Uses global reference ranges for absolute normalization.
    Includes ocean acidification (pH) and deoxygenation (O₂).

    Terrestrial components (all populations):
      - Temperature extremes: bio5 (max temp warmest month)
      - Temperature variability: bio7 (annual temp range)
      - Seasonality: bio4 (temperature seasonality)

    Marine components (where available):
      - Thermal stress: sst_max (max sea surface temperature)
      - Thermal variability: sst_range (annual SST range)
      - Ocean acidification: inverted ph_mean (lower pH = more stress)
      - Deoxygenation: inverted o2_mean (lower O₂ = more stress)

    Weighting depends on whether marine data is available.
    """
    exposure = pd.DataFrame(index=df.index)

    # ── Terrestrial stressors (available for all) ──
    exposure["temp_extreme"] = norm01_abs(df["bio5"], "bio5") if "bio5" in df.columns else 0.5
    exposure["temp_variability"] = norm01_abs(df["bio7"], "bio7") if "bio7" in df.columns else 0.5
    exposure["seasonality"] = norm01_abs(df["bio4"], "bio4") if "bio4" in df.columns else 0.5

    # ── Marine stressors (California only) ──
    has_marine = "sst_max" in df.columns and df["sst_max"].notna().any()

    if has_marine:
        exposure["sst_stress"] = norm01_abs(
            df["sst_max"].fillna(df["sst_max"].median()), "sst_max"
        )
        exposure["sst_variability"] = norm01_abs(
            df["sst_range"].fillna(df["sst_range"].median()), "sst_range"
        ) if "sst_range" in df.columns else 0.5

        # Ocean acidification: LOWER pH = MORE stress → invert
        if "ph_mean" in df.columns and df["ph_mean"].notna().any():
            exposure["acidification"] = 1 - norm01_abs(
                df["ph_mean"].fillna(df["ph_mean"].median()), "ph_mean"
            )
        else:
            exposure["acidification"] = 0.5

        # Deoxygenation: LOWER O₂ = MORE stress → invert
        if "o2_mean" in df.columns and df["o2_mean"].notna().any():
            exposure["deoxygenation"] = 1 - norm01_abs(
                df["o2_mean"].fillna(df["o2_mean"].median()), "o2_mean"
            )
        else:
            exposure["deoxygenation"] = 0.5

        # Marine-weighted composite
        exposure_index = (
            # Terrestrial (30% total)
            0.10 * exposure["temp_extreme"]
            + 0.10 * exposure["temp_variability"]
            + 0.10 * exposure["seasonality"]
            # Marine (70% total)
            + 0.25 * exposure["sst_stress"]
            + 0.15 * exposure["sst_variability"]
            + 0.15 * exposure["acidification"]
            + 0.15 * exposure["deoxygenation"]
        )
    else:
        # Terrestrial-only composite
        exposure_index = (
            0.40 * exposure["temp_extreme"]
            + 0.30 * exposure["temp_variability"]
            + 0.30 * exposure["seasonality"]
        )

    return exposure_index


# ── The pipeline's definition of exposure ──
# Exposure = normalized magnitude of the projected CHANGE in climate parameters
# between the historical baseline and SSP5-8.5 end of century (climate_delta.py),
# NOT the absolute value of those parameters in any single scenario slice.
compute_climate_exposure = climate_delta.compute_climate_exposure
GCMS = climate_delta.GCMS

# Weight of coastal land-use pressure (NLCD 2021, 30 m) in the combined stressor.
# Land use is not climate, so it is kept as its own term rather than folded into
# the climate index; points outside NLCD coverage fall back to climate only.
LULC_WEIGHT = 0.25


def classify_resilience(diversity_norm, exposure, sigma_norm):
    """Classify populations into resilience quadrants.

    Returns: category label and color
    """
    # Thresholds: median-split for diversity and exposure
    div_high = diversity_norm >= 0.5
    exp_high = exposure >= 0.5

    labels = np.where(
        div_high & ~exp_high, "Resilient",
        np.where(
            div_high & exp_high, "At Risk",
            np.where(
                ~div_high & ~exp_high, "Latent Vulnerability",
                "Critical"
            )
        )
    )

    # Upgrade to Critical if uncertainty is very high (precautionary)
    high_uncertainty = sigma_norm >= 0.7
    labels = np.where(high_uncertainty & (labels == "Latent Vulnerability"),
                      "Critical", labels)

    return labels


def main():
    model, processor = load_model_and_processor()
    cali_df = pd.read_csv(PROC / _p.CALI_FILE)
    cali_df = _m.filter_ca_species(cali_df)
    coast_lats, coast_lons = make_coastal_grid()

    species_info = cali_df.groupby("species").agg(
        metric_type=("metric_type", "first"),
        n_pops=("gen_div", "count"),
        mean_div=("gen_div", "mean"),
    ).reset_index()

    # ── Predict for all species along coast ──
    # Each species is restricted to its plausible latitudinal range
    # (observed range ± buffer) so that exposure scores are species-specific.
    LAT_BUFFER_DEG = 1.5  # degrees beyond observed range

    print("Generating predictions for all species...")
    results = []

    for _, sp_row in species_info.iterrows():
        sp = sp_row["species"]
        mt = sp_row["metric_type"]
        obs = cali_df[cali_df["species"] == sp]

        if len(obs) < 2:
            continue

        # Species-specific latitude range
        obs_lat_min = obs["Latitude"].min() - LAT_BUFFER_DEG
        obs_lat_max = obs["Latitude"].max() + LAT_BUFFER_DEG
        range_mask = (coast_lats >= obs_lat_min) & (coast_lats <= obs_lat_max)

        # Ensure at least 10 grid points for meaningful predictions
        if range_mask.sum() < 10:
            center = obs["Latitude"].mean()
            half_span = (coast_lats[-1] - coast_lats[0]) / 150 * 5  # ~5 grid spacings
            range_mask = (coast_lats >= center - half_span) & (coast_lats <= center + half_span)

        sp_lats = coast_lats[range_mask]
        sp_lons = coast_lons[range_mask]

        context = prepare_context(obs, processor)
        target_df = make_target_df(sp_lats, sp_lons, sp, mt, cali_df)
        mu, sigma = predict_at_locations(model, context, target_df, processor)

        # Exposure at each point: the change-based index, plus the two
        # superseded state-based indices for the comparison figures
        exposure_v1 = compute_climate_exposure_v1(target_df)
        exposure_state = compute_climate_exposure_state(target_df)
        exposure_change = compute_climate_exposure(target_df)

        for i in range(len(sp_lats)):
            results.append({
                "species": sp,
                "metric_type": mt,
                "Latitude": sp_lats[i],
                "Longitude": sp_lons[i],
                "pred_mu": mu[i],
                "pred_sigma": sigma[i],
                "climate_exposure_v1": exposure_v1.iloc[i],
                "climate_exposure_state": exposure_state.iloc[i],
                "climate_exposure": exposure_change.iloc[i],
                "n_obs": len(obs),
                "is_expert": bool(obs["is_expert"].any()) if "is_expert" in obs.columns else False,
            })

        print(f"  {sp}: lat [{obs_lat_min+LAT_BUFFER_DEG:.1f}, {obs_lat_max-LAT_BUFFER_DEG:.1f}] "
              f"→ grid [{sp_lats[0]:.1f}, {sp_lats[-1]:.1f}] ({len(sp_lats)} pts)")

    results_df = pd.DataFrame(results)

    # ── Inter-model spread of the change signal ──
    print("Computing per-GCM exposure (climate change ensemble)...")
    e_mean, e_sd, e_min, e_max, per_gcm = climate_delta.exposure_ensemble(results_df)
    results_df["climate_exposure"] = e_mean
    results_df["climate_exposure_sd"] = e_sd
    results_df["climate_exposure_min"] = e_min
    results_df["climate_exposure_max"] = e_max
    for g, vals in per_gcm.items():
        results_df[f"climate_exposure_{g}"] = vals
    print(f"  ensemble mean {e_mean.mean():.4f}, inter-model sd {e_sd.mean():.4f}")

    # ── Load global data for reference ranges ──
    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)

    # Global He reference (2nd–98th percentile to exclude outliers)
    HE_REF_LO = global_df["gen_div"].quantile(0.02)
    HE_REF_HI = global_df["gen_div"].quantile(0.98)
    # California π reference
    pi_vals = cali_df[cali_df["metric_type"] == "pi"]["gen_div"]
    PI_REF_LO = pi_vals.quantile(0.02)
    PI_REF_HI = pi_vals.quantile(0.98)
    print(f"  He global reference: [{HE_REF_LO:.4f}, {HE_REF_HI:.4f}]")
    print(f"  π California reference: [{PI_REF_LO:.6f}, {PI_REF_HI:.6f}]")

    # ── Coastal land-use pressure (non-climate stressor) ──
    print("Adding NLCD land-use pressure...")
    lu = land_use.pressure_at(results_df["Latitude"].values,
                              results_df["Longitude"].values)
    for c in ["lulc_pressure", "natural_frac", "developed_frac", "crop_frac", "land_frac"]:
        results_df[c] = lu[c].values
    results_df["lulc_available"] = ~lu["outside_nlcd"].values & lu["lulc_pressure"].notna().values

    have = results_df["lulc_available"].values
    results_df["total_stress"] = results_df["climate_exposure"]
    results_df.loc[have, "total_stress"] = (
        (1 - LULC_WEIGHT) * results_df.loc[have, "climate_exposure"]
        + LULC_WEIGHT * results_df.loc[have, "lulc_pressure"]
    )
    print(f"  land-use pressure available for {have.sum()}/{len(results_df)} predictions "
          f"(mean {results_df.loc[have, 'lulc_pressure'].mean():.3f})")
    print(f"  combined stressor: {results_df['total_stress'].mean():.4f} "
          f"vs climate alone {results_df['climate_exposure'].mean():.4f}")

    # ── Compute vulnerability scores ──
    print("Computing vulnerability scores...")

    # --- Within-species normalization: FOR PLOTTING ONLY ---
    # diversity_norm_within rescales each species onto its own 0-1 range. It is a
    # display transform, not the score: leave-one-out shows the model captures
    # only 13% (He) and 1% (pi) of within-species variance, with a median
    # within-species correlation of 0.01, so min-max stretching that component
    # would amplify noise into apparent structure.
    #
    # The score below uses the cross-species reference instead, which rests on
    # the between-species signal the model does resolve. Predictions themselves
    # stay in raw He / pi units throughout (pred_mu).
    # If 06c_future_diversity.py has already run, its scale spans present AND
    # future predictions, so the two slices share one within-species axis. On a
    # first pass that file does not exist yet and the present-day spatial range
    # is used to bootstrap; rerun this step after 06c to settle on the joint scale.
    scale_path = PROC / "diversity_scaling.csv"
    joint = (pd.read_csv(scale_path).set_index("species")
             if scale_path.exists() else None)
    print("  diversity scale: " + ("present ∪ future (from 06c)" if joint is not None
                                   else "present only (bootstrap — rerun after 06c)"))

    scaling = []
    for sp in results_df["species"].unique():
        mask = results_df["species"] == sp
        vals = results_df.loc[mask, "pred_mu"]
        if joint is not None and sp in joint.index:
            mn, mx = joint.loc[sp, "div_lo"], joint.loc[sp, "div_hi"]
        else:
            mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-8:
            results_df.loc[mask, "diversity_norm_within"] = ((vals - mn) / (mx - mn)).clip(0, 1)
        else:
            results_df.loc[mask, "diversity_norm_within"] = 0.5
        scaling.append({"species": sp,
                        "metric_type": results_df.loc[mask, "metric_type"].iloc[0],
                        "div_lo": mn, "div_hi": mx})

        svals = results_df.loc[mask, "pred_sigma"]
        smn, smx = svals.min(), svals.max()
        if smx - smn > 1e-8:
            results_df.loc[mask, "sigma_norm"] = (svals - smn) / (smx - smn)
        else:
            results_df.loc[mask, "sigma_norm"] = 0.0

    if joint is None:
        pd.DataFrame(scaling).to_csv(scale_path, index=False)
        print(f"  saved bootstrap diversity scale: {scale_path}")

    # --- Cross-species reference normalization, kept for comparison figures ---
    he_mask = results_df["metric_type"] == "He"
    pi_mask = results_df["metric_type"] == "pi"
    results_df.loc[he_mask, "diversity_norm"] = (
        (results_df.loc[he_mask, "pred_mu"] - HE_REF_LO) / (HE_REF_HI - HE_REF_LO)
    ).clip(0, 1)
    results_df.loc[pi_mask, "diversity_norm"] = (
        (results_df.loc[pi_mask, "pred_mu"] - PI_REF_LO) / (PI_REF_HI - PI_REF_LO)
    ).clip(0, 1)
    # aliases kept for older downstream code
    results_df["diversity_norm_global"] = results_df["diversity_norm"]
    results_df["diversity_norm_ws"] = results_df["diversity_norm_within"]

    # Vulnerability = (1 - within-species diversity) × combined stressor + 0.3σ,
    # where the combined stressor is Δ-climate exposure plus land-use pressure.
    results_df["vulnerability"] = (
        (1 - results_df["diversity_norm_within"]) * results_df["total_stress"]
        + 0.3 * results_df["sigma_norm"]
    )
    results_df["vulnerability_climate_only"] = (
        (1 - results_df["diversity_norm_within"]) * results_df["climate_exposure"]
        + 0.3 * results_df["sigma_norm"]
    )

    results_df["vulnerability_lo"] = (
        (1 - results_df["diversity_norm_within"]) * results_df["climate_exposure_min"]
        + 0.3 * results_df["sigma_norm"]
    )
    results_df["vulnerability_hi"] = (
        (1 - results_df["diversity_norm_within"]) * results_df["climate_exposure_max"]
        + 0.3 * results_df["sigma_norm"]
    )

    # Resilience classification (within-species diversity position)
    results_df["resilience_class"] = classify_resilience(
        results_df["diversity_norm_within"].values,
        results_df["total_stress"].values,
        results_df["sigma_norm"].values,
    )

    # Superseded version for comparison: state exposure + cross-species diversity
    results_df["vulnerability_old"] = (
        (1 - results_df["diversity_norm_global"]) * results_df["climate_exposure_v1"]
        + 0.3 * results_df["sigma_norm"]
    )
    results_df["resilience_class_old"] = classify_resilience(
        results_df["diversity_norm_global"].values,
        results_df["climate_exposure_v1"].values,
        results_df["sigma_norm"].values,
    )

    results_df.to_csv(PROC / "vulnerability_scores.csv", index=False)
    print(f"Saved: {PROC / 'vulnerability_scores.csv'} ({len(results_df)} records)")

    # ── Shared basemap setup ──
    import geopandas as gpd
    import ca_basemap as _cabm
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    from pyproj import Transformer

    CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
    CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ca_xmin, ca_ymin = transformer.transform(CA_LON_MIN, CA_LAT_MIN)
    ca_xmax, ca_ymax = transformer.transform(CA_LON_MAX, CA_LAT_MAX)

    top6 = species_info.nlargest(6, "n_pops")

    def _setup_ca_panel(ax):
        """Set fixed California extent and draw the offline land basemap."""
        _cabm.add_land(ax, ca_xmin, ca_xmax, ca_ymin, ca_ymax)

    # ── Figure 1: Resilience map with California basemap ──
    color_map = {
        "Resilient": "#2ecc71",
        "At Risk": "#f1c40f",
        "Latent Vulnerability": "#e67e22",
        "Critical": "#e74c3c",
    }

    fig = plt.figure(figsize=(24, 32))
    gs = GridSpec(3, 3, figure=fig, hspace=0.08, wspace=0.05,
                  top=0.93, bottom=0.06, left=0.02, right=0.98)

    for idx, (_, sp_row) in enumerate(top6.iterrows()):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sp = sp_row["species"]
        sp_data = results_df[results_df["species"] == sp]

        _setup_ca_panel(ax)

        gdf = gpd.GeoDataFrame(
            sp_data,
            geometry=gpd.points_from_xy(sp_data["Longitude"], sp_data["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        for cls, color in color_map.items():
            mask = gdf["resilience_class"] == cls
            if mask.sum() > 0:
                gdf[mask].plot(
                    ax=ax, color=color, markersize=60, alpha=0.75,
                    edgecolors="black", linewidth=0.3, zorder=3,
                )

        obs = cali_df[cali_df["species"] == sp]
        obs_gdf = gpd.GeoDataFrame(
            obs,
            geometry=gpd.points_from_xy(obs["Longitude"], obs["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        obs_gdf.plot(ax=ax, color="black", marker="^", markersize=90,
                     zorder=4, edgecolors="white", linewidth=0.8)

        ax.set_xlim(ca_xmin, ca_xmax)
        ax.set_ylim(ca_ymin, ca_ymax)
        ax.set_title(f"{sp}\n({sp_row['metric_type']}, n={sp_row['n_pops']})",
                     fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend in the third row (spanning all 3 columns)
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=22, markeredgecolor="black", markeredgewidth=0.5,
               label="Resilient\n(High diversity, Low exposure)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f",
               markersize=22, markeredgecolor="black", markeredgewidth=0.5,
               label="At Risk\n(High diversity, High exposure)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22",
               markersize=22, markeredgecolor="black", markeredgewidth=0.5,
               label="Latent Vulnerability\n(Low diversity, Low exposure)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=22, markeredgecolor="black", markeredgewidth=0.5,
               label="Critical\n(Low diversity, High exposure)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               markersize=20, markeredgecolor="white", markeredgewidth=0.8,
               label="Observed population"),
    ]
    ax_leg.legend(
        handles=legend_elements,
        loc="center",
        ncol=5,
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    fig.suptitle("Climate Change Resilience Classification Along California Coast",
                 fontsize=20, fontweight="bold")
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"resilience_map.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'resilience_map.png'}")

    # ── Figure 1b: Predicted diversity map (same 6 species, same layout) ──
    fig = plt.figure(figsize=(24, 32))
    gs = GridSpec(3, 3, figure=fig, hspace=0.08, wspace=0.05,
                  top=0.93, bottom=0.06, left=0.02, right=0.98)

    for idx, (_, sp_row) in enumerate(top6.iterrows()):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sp = sp_row["species"]
        mt = sp_row["metric_type"]
        sp_data = results_df[results_df["species"] == sp]

        _setup_ca_panel(ax)

        # Predictions colored by predicted diversity (pred_mu)
        gdf = gpd.GeoDataFrame(
            sp_data,
            geometry=gpd.points_from_xy(sp_data["Longitude"], sp_data["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        # Set color range per species (include observed values for full range)
        obs_vals = cali_df[cali_df["species"] == sp]["gen_div"]
        all_vals = np.concatenate([sp_data["pred_mu"].values, obs_vals.values])
        vmin = all_vals.min() - (all_vals.max() - all_vals.min()) * 0.1
        vmax = all_vals.max() + (all_vals.max() - all_vals.min()) * 0.1
        cbar_label = f"Predicted {'He' if mt == 'He' else 'π'}"

        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=sp_data["pred_mu"].values, cmap="viridis",
            s=60, alpha=0.8, edgecolors="black", linewidth=0.3,
            vmin=vmin, vmax=vmax, zorder=3,
        )

        # Overlay observed points colored by actual diversity
        obs = cali_df[cali_df["species"] == sp]
        obs_gdf = gpd.GeoDataFrame(
            obs,
            geometry=gpd.points_from_xy(obs["Longitude"], obs["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        ax.scatter(
            obs_gdf.geometry.x, obs_gdf.geometry.y,
            c=obs["gen_div"].values, cmap="viridis",
            s=100, marker="^", edgecolors="red", linewidth=1.5,
            vmin=vmin, vmax=vmax, zorder=4,
        )

        ax.set_xlim(ca_xmin, ca_xmax)
        ax.set_ylim(ca_ymin, ca_ymax)

        # Per-panel colorbar
        cb = plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.01)
        cb.set_label(cbar_label, fontsize=10)
        cb.ax.tick_params(labelsize=9)

        # Add mean ± std in title
        mean_div = sp_data["pred_mu"].mean()
        std_div = sp_data["pred_mu"].std()
        ax.set_title(f"{sp}\n({mt}, n={sp_row['n_pops']}, "
                     f"mean={mean_div:.3f} ± {std_div:.3f})",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend in third row
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    div_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#21918c",
               markersize=20, markeredgecolor="black", markeredgewidth=0.5,
               label="Predicted diversity\n(circles along coast)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#21918c",
               markersize=20, markeredgecolor="red", markeredgewidth=1.5,
               label="Observed diversity\n(triangles, red-edged)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#fde725",
               markersize=18, markeredgecolor="black", markeredgewidth=0.5,
               label="High diversity"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#440154",
               markersize=18, markeredgecolor="black", markeredgewidth=0.5,
               label="Low diversity"),
    ]
    ax_leg.legend(
        handles=div_legend,
        loc="center",
        ncol=4,
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    fig.suptitle("Predicted Genetic Diversity Along California Coast",
                 fontsize=20, fontweight="bold")
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"diversity_map.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'diversity_map.png'}")

    # ── Figure 2: Species vulnerability ranking ──
    sp_vuln = results_df.groupby("species").agg(
        mean_vulnerability=("vulnerability", "mean"),
        mean_exposure=("climate_exposure", "mean"),
        mean_diversity=("diversity_norm", "mean"),
        mean_uncertainty=("sigma_norm", "mean"),
        pct_critical=("resilience_class", lambda x: (x == "Critical").mean()),
        metric_type=("metric_type", "first"),
        n_obs=("n_obs", "first"),
    ).reset_index().sort_values("mean_vulnerability", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#e74c3c" if row["pct_critical"] > 0.5 else
              "#e67e22" if row["pct_critical"] > 0.3 else
              "#f1c40f" if row["pct_critical"] > 0.1 else
              "#2ecc71" for _, row in sp_vuln.iterrows()]

    bars = ax.barh(range(len(sp_vuln)), sp_vuln["mean_vulnerability"],
                   color=colors, alpha=0.8)

    ax.set_yticks(range(len(sp_vuln)))
    labels = [f"{row['species']} ({row['metric_type']}, n={row['n_obs']:.0f})"
              for _, row in sp_vuln.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Vulnerability Score")
    ax.set_title("Species Vulnerability Ranking\n"
                 "(Green=Resilient, Yellow=At Risk, Orange=Vulnerable, Red=>50% Critical)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"species_vulnerability_ranking.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'species_vulnerability_ranking.png'}")

    # ── Figure 3: Full 4-way comparison ──
    # Rows: He, π
    # Cols: OLD (within-species div + local exposure), NEW (global-ref div + marine exposure)
    he_data = results_df[results_df["metric_type"] == "He"]
    pi_data = results_df[results_df["metric_type"] == "pi"]

    def _plot_quadrant(ax, data, exp_col, div_col, title, xlabel, ylabel):
        for sp in data["species"].unique():
            sp_d = data[data["species"] == sp]
            mean_div = sp_d[div_col].mean()
            mean_exp = sp_d[exp_col].mean()
            mean_unc = sp_d["sigma_norm"].mean()
            ax.scatter(mean_exp, mean_div, s=100 + mean_unc * 200,
                       alpha=0.7, label=sp[:22], edgecolors="black", linewidth=0.3)
        for txt, x, y, c in [("Resilient",0.25,0.75,"#2ecc71"),
                               ("At Risk",0.75,0.75,"#f1c40f"),
                               ("Latent\nVulnerable",0.25,0.25,"#e67e22"),
                               ("Critical",0.75,0.25,"#e74c3c")]:
            ax.text(x, y, txt, ha="center", fontsize=10, color=c, alpha=0.5)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
        ax.axvline(0.5, color="gray", ls="--", alpha=0.3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=5, ncol=2, loc="best")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 4, figsize=(28, 14))

    # Row 1: He
    _plot_quadrant(axes[0, 0], he_data, "climate_exposure_v1", "diversity_norm",
                   "He: OLD exposure + OLD diversity\n(local norm, no pH/O₂, within-species div)",
                   "Exposure V1 (local)", "He diversity (within-species)")
    _plot_quadrant(axes[0, 1], he_data, "climate_exposure", "diversity_norm",
                   "He: CHANGE exposure + within-species diversity\n(current definition)",
                   "Δ-climate exposure (change)", "He diversity (within-species)")
    _plot_quadrant(axes[0, 2], he_data, "climate_exposure_v1", "diversity_norm_global",
                   "He: OLD exposure + NEW diversity\n(local norm, global He reference)",
                   "Exposure V1 (local)", "He diversity (global ref)")
    _plot_quadrant(axes[0, 3], he_data, "climate_exposure", "diversity_norm_global",
                   "He: CHANGE exposure + cross-species diversity\n(Δ climate + global He reference)",
                   "Δ-climate exposure (change)", "He diversity (global ref)")

    # Row 2: π
    _plot_quadrant(axes[1, 0], pi_data, "climate_exposure_v1", "diversity_norm",
                   "π: OLD exposure + OLD diversity\n(local norm, no pH/O₂, within-species div)",
                   "Exposure V1 (local)", "π diversity (within-species)")
    _plot_quadrant(axes[1, 1], pi_data, "climate_exposure", "diversity_norm",
                   "π: CHANGE exposure + within-species diversity\n(current definition)",
                   "Δ-climate exposure (change)", "π diversity (within-species)")
    _plot_quadrant(axes[1, 2], pi_data, "climate_exposure_v1", "diversity_norm_global",
                   "π: OLD exposure + NEW diversity\n(local norm, CA π reference)",
                   "Exposure V1 (local)", "π diversity (CA ref)")
    _plot_quadrant(axes[1, 3], pi_data, "climate_exposure", "diversity_norm_global",
                   "π: CHANGE exposure + cross-species diversity\n(Δ climate + CA π reference)",
                   "Δ-climate exposure (change)", "π diversity (CA ref)")

    plt.suptitle("Diversity vs Exposure: 4-Way Comparison\n"
                 "Left→Right: state exposure → change-based exposure | Top: He, Bottom: π\n"
                 "Column 4 (rightmost) = final recommended approach",
                 fontsize=14, y=1.03)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"diversity_vs_exposure_comparison.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'diversity_vs_exposure_comparison.png'}")

    # ── Figure 3b: Resilience class shift (old vs new) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: Sankey-style shift table
    ax = axes[0]
    classes = ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    old_pcts = [(results_df["resilience_class_old"] == c).mean() * 100 for c in classes]
    new_pcts = [(results_df["resilience_class"] == c).mean() * 100 for c in classes]
    x = np.arange(len(classes))
    w = 0.35
    bars1 = ax.bar(x - w/2, old_pcts, w, label="OLD (state exposure + cross-species div)",
                    color=colors, alpha=0.5, edgecolor="black")
    bars2 = ax.bar(x + w/2, new_pcts, w, label="NEW (Δ-climate exposure + within-species div)",
                    color=colors, alpha=0.9, edgecolor="black")
    for bar, pct in zip(bars1, old_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=8, color="gray")
    for bar, pct in zip(bars2, new_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("% of predictions")
    ax.set_title("Resilience Classification Shift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel B: Species vulnerability ranking comparison
    ax = axes[1]
    sp_old = results_df.groupby("species")["vulnerability_old"].mean().sort_values()
    sp_new = results_df.groupby("species")["vulnerability"].mean()
    sp_list = sp_old.index.tolist()
    y_pos = range(len(sp_list))
    ax.barh(y_pos, [sp_old[sp] for sp in sp_list], height=0.4, alpha=0.4,
            color="gray", label="OLD")
    ax.barh([y + 0.4 for y in y_pos], [sp_new[sp] for sp in sp_list],
            height=0.4, alpha=0.8, color="steelblue", label="NEW")
    ax.set_yticks([y + 0.2 for y in y_pos])
    mt_map = results_df.groupby("species")["metric_type"].first()
    ax.set_yticklabels([f"{sp[:22]} ({mt_map[sp]})" for sp in sp_list], fontsize=6)
    ax.set_xlabel("Mean Vulnerability Score")
    ax.set_title("Species Vulnerability: OLD vs NEW")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # Panel C: Exposure V1 vs V2 scatter
    ax = axes[2]
    ax.scatter(results_df["climate_exposure_state"], results_df["climate_exposure"],
               c=results_df["Latitude"], cmap="coolwarm", s=5, alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("State exposure (absolute climate, superseded)")
    ax.set_ylabel("Δ-climate exposure (projected change)")
    ax.set_title("State vs change-based exposure\n(colored by latitude: blue=north, red=south)")
    cb = plt.colorbar(ax.collections[0], ax=ax, label="Latitude", shrink=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.suptitle("Impact of Defining Exposure as Projected CHANGE + Global Reference Normalization",
                 fontsize=13)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"old_vs_new_comparison.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'old_vs_new_comparison.png'}")

    # ── Figure 3c: Exposure component breakdown ──
    # Uses full coast to show how components vary spatially
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    sample_sp = species_info.iloc[0]["species"]
    sample_mt = species_info.iloc[0]["metric_type"]
    full_target = make_target_df(coast_lats, coast_lons, sample_sp, sample_mt, cali_df)

    # V1 components
    ax = axes[0]
    v1_comps = {}
    v1_comps["temp_stress"] = full_target["bio5"] if "bio5" in full_target.columns else pd.Series(0)
    v1_comps["seasonality"] = full_target["bio4"] if "bio4" in full_target.columns else pd.Series(0)
    v1_comps["precip_stress"] = full_target["bio14"] if "bio14" in full_target.columns else pd.Series(0)
    v1_comps["sst_max"] = full_target["sst_max"] if "sst_max" in full_target.columns else pd.Series(0)
    for name, vals in v1_comps.items():
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-8:
            normed = (vals - mn) / (mx - mn)
        else:
            normed = pd.Series(0.5, index=vals.index)
        if name == "precip_stress":
            normed = 1 - normed
        ax.plot(coast_lats, normed, label=name, alpha=0.7)
    ax.plot(coast_lats, compute_climate_exposure_v1(full_target),
            "k-", linewidth=2.5, label="V1 composite", alpha=0.9)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Normalized stress (0–1)")
    ax.set_title(f"V1 Components ({sample_sp})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Change-based components, with per-component inter-GCM min–max bands.
    # Marine components share one Bio-ORACLE SSP5-8.5 projection, so their band is
    # zero-width; the terrestrial components (bio*) carry the GCM spread.
    ax = axes[1]
    _, comp, _ = compute_climate_exposure(full_target, return_components=True)
    per_gcm_comp = {g: compute_climate_exposure(full_target, gcm=g, return_components=True)[1]
                    for g in climate_delta.GCMS}
    labels = {
        "d_sst_mean": "Δ SST mean", "d_sst_max": "Δ SST max",
        "d_sst_range": "|Δ SST range|", "d_ph": "acidification (−Δ pH)",
        "d_o2": "deoxygenation (−Δ O₂)", "sst_novelty": "thermal novelty",
        "d_bio5": "Δ bio5", "d_bio1": "Δ bio1", "d_bio4": "|Δ bio4|",
        "dry_bio14": "drying (bio14)",
    }
    for key, label in labels.items():
        if key in comp.columns:
            line, = ax.plot(coast_lats, comp[key], label=label, alpha=0.85, lw=1.3)
            stack = np.column_stack([per_gcm_comp[g][key].values for g in climate_delta.GCMS])
            ax.fill_between(coast_lats, stack.min(axis=1), stack.max(axis=1),
                            color=line.get_color(), alpha=0.18)
    ax.plot(coast_lats, compute_climate_exposure(full_target),
            "k-", linewidth=2.5, label="Δ-climate composite", alpha=0.9)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Normalized change (0–1)")
    ax.set_title(f"Change-based components ({sample_sp})\n(shaded = inter-GCM min–max; marine shared)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # GCM variability of the Δ-climate composite exposure
    ax = axes[2]
    per_gcm = {g: climate_delta.compute_climate_exposure(full_target, gcm=g).values
               for g in climate_delta.GCMS}
    mat = np.column_stack([per_gcm[g] for g in climate_delta.GCMS])
    ens = climate_delta.compute_climate_exposure(full_target).values
    ax.fill_between(coast_lats, mat.min(axis=1), mat.max(axis=1),
                    color="grey", alpha=0.25, label="GCM range (min–max)")
    for g in climate_delta.GCMS:
        ax.plot(coast_lats, per_gcm[g], lw=1.0, alpha=0.8, label=g)
    ax.plot(coast_lats, ens, "k-", lw=2.6, label="Ensemble mean")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Δ-climate exposure (0–1)")
    ax.set_title(f"GCM variability of Δ-exposure ({sample_sp})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Exposure Components Along the California Coast: "
                 "superseded state index, projected change, and inter-GCM spread", fontsize=13)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"exposure_components.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'exposure_components.png'}")

    # ── Figure 4: Global vs California comparison ──

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: He distribution comparison
    ax = axes[0]
    ax.hist(global_df["gen_div"], bins=50, alpha=0.5, label="Global (GenDivRange)",
            density=True, color="steelblue")
    cali_he = cali_df[cali_df["metric_type"] == "He"]["gen_div"]
    if len(cali_he) > 0:
        ax.hist(cali_he, bins=20, alpha=0.5, label="California He",
                density=True, color="coral")
    ax.set_xlabel("Genetic Diversity (He)")
    ax.set_ylabel("Density")
    ax.set_title("He Distribution: Global vs California")
    ax.legend()

    # Panel B: Diversity vs latitude
    ax = axes[1]
    ax.scatter(global_df["abs_latitude"], global_df["gen_div"],
               s=2, alpha=0.1, color="steelblue", label="Global")
    ax.scatter(cali_df["abs_latitude"], cali_df["gen_div"],
               s=20, alpha=0.7, color="coral", label="California")
    ax.set_xlabel("Absolute Latitude")
    ax.set_ylabel("Genetic Diversity")
    ax.set_title("Latitudinal Diversity Gradient")
    ax.legend()

    # Panel C: Climate stress vs diversity
    ax = axes[2]
    if "climate_stress" in global_df.columns:
        ax.scatter(global_df["climate_stress"], global_df["gen_div"],
                   s=2, alpha=0.1, color="steelblue", label="Global")
    if "climate_stress" in cali_df.columns:
        ax.scatter(cali_df["climate_stress"], cali_df["gen_div"],
                   s=20, alpha=0.7, color="coral", label="California")
    ax.set_xlabel("Climate Stress Index")
    ax.set_ylabel("Genetic Diversity")
    ax.set_title("Climate Stress vs Diversity")
    ax.legend()

    plt.suptitle("Global vs California Genetic Diversity Patterns", fontsize=13)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"global_vs_california.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'global_vs_california.png'}")

    # ── Figure 5: Ecosystem-level resilience (all species aggregated) ──
    print("Generating ecosystem-level resilience map...")
    import geopandas as gpd
    import ca_basemap as _cabm
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    from pyproj import Transformer

    # Bin predictions by latitude (same coastal grid)
    # For each latitude bin, aggregate across ALL species present at that location
    lat_bin_edges = np.linspace(coast_lats.min(), coast_lats.max(), 76)  # ~0.15° bins
    lat_bin_centers = (lat_bin_edges[:-1] + lat_bin_edges[1:]) / 2

    eco_stats = []
    for i in range(len(lat_bin_edges) - 1):
        lo, hi = lat_bin_edges[i], lat_bin_edges[i + 1]
        mask = (results_df["Latitude"] >= lo) & (results_df["Latitude"] < hi)
        pts = results_df[mask]
        if len(pts) == 0:
            continue

        n_species = pts["species"].nunique()
        n_critical = (pts["resilience_class"] == "Critical").sum()
        n_at_risk = (pts["resilience_class"] == "At Risk").sum()
        n_resilient = (pts["resilience_class"] == "Resilient").sum()
        n_latent = (pts["resilience_class"] == "Latent Vulnerability").sum()
        n_total = len(pts)

        # Interpolate longitude for this latitude from coastal grid
        lon_center = np.interp(lat_bin_centers[i], coast_lats, coast_lons)

        eco_stats.append({
            "lat_center": lat_bin_centers[i],
            "lon_center": lon_center,
            "n_species": n_species,
            "mean_vulnerability": pts["vulnerability"].mean(),
            "median_vulnerability": pts["vulnerability"].median(),
            "mean_diversity": pts["diversity_norm"].mean(),
            "mean_exposure": pts["climate_exposure"].mean(),
            "mean_uncertainty": pts["sigma_norm"].mean(),
            "pct_critical": n_critical / n_total,
            "pct_at_risk": n_at_risk / n_total,
            "pct_resilient": n_resilient / n_total,
            "pct_latent": n_latent / n_total,
            "species_list": sorted(pts["species"].unique()),
        })

    eco_df = pd.DataFrame(eco_stats)
    print(f"  {len(eco_df)} coastal segments, "
          f"{results_df['species'].nunique()} species total")

    # --- Panel layout: 4 subplots ---
    fig = plt.figure(figsize=(28, 24))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.25, wspace=0.2)

    # ─── Panel A: Ecosystem resilience map (basemap) ───
    ax_map = fig.add_subplot(gs[:, 0])

    CA_LON_MIN, CA_LON_MAX = -125.5, -114.5
    CA_LAT_MIN, CA_LAT_MAX = 29.5, 43.0
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ca_xmin, ca_ymin = transformer.transform(CA_LON_MIN, CA_LAT_MIN)
    ca_xmax, ca_ymax = transformer.transform(CA_LON_MAX, CA_LAT_MAX)

    _cabm.add_land(ax_map, ca_xmin, ca_xmax, ca_ymin, ca_ymax)

    # Color by mean vulnerability
    vuln_cmap = LinearSegmentedColormap.from_list(
        "vuln", ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    )
    eco_gdf = gpd.GeoDataFrame(
        eco_df,
        geometry=gpd.points_from_xy(eco_df["lon_center"], eco_df["lat_center"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Scale point size by number of species
    sizes = eco_df["n_species"] / eco_df["n_species"].max() * 200 + 50

    sc = ax_map.scatter(
        eco_gdf.geometry.x, eco_gdf.geometry.y,
        c=eco_df["median_vulnerability"], cmap=vuln_cmap,
        s=sizes, alpha=0.85,
        edgecolors="black", linewidth=0.4,
        vmin=0.1, vmax=0.7, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax_map, shrink=0.5, pad=0.02,
                       label="Median Vulnerability Score")
    cb.ax.tick_params(labelsize=11)
    cb.set_label("Mean Vulnerability Score", fontsize=13)

    ax_map.set_xlim(ca_xmin, ca_xmax)
    ax_map.set_ylim(ca_ymin, ca_ymax)
    ax_map.set_title("Ecosystem Resilience\n(All Species Aggregated)",
                     fontsize=16, fontweight="bold")
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    # Add note about point size
    for ns, label in [(5, "5 spp"), (15, "15 spp"), (25, "25 spp")]:
        s = ns / eco_df["n_species"].max() * 200 + 50
        ax_map.scatter([], [], s=s, c="gray", alpha=0.6, edgecolors="black",
                       linewidth=0.4, label=label)
    ax_map.legend(title="Species coverage", loc="lower left", fontsize=11,
                  title_fontsize=12, frameon=True, fancybox=True)

    # ─── Panel B: Coastal transect — stacked resilience classes ───
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

    # Overlay species count
    ax_count = ax_stack.twinx()
    ax_count.plot(eco_df["lat_center"], eco_df["n_species"],
                  "k-", linewidth=2, alpha=0.6)
    ax_count.set_ylabel("Number of species", fontsize=12, color="black")
    ax_count.tick_params(axis="y", labelsize=10)

    ax_stack.set_xlabel("Latitude", fontsize=12)
    ax_stack.set_ylabel("Proportion of species", fontsize=12)
    ax_stack.set_title("Resilience Class Composition Along Coast",
                       fontsize=14, fontweight="bold")
    ax_stack.set_ylim([0, 1])
    ax_stack.legend(loc="upper right", fontsize=10, frameon=True)
    ax_stack.tick_params(labelsize=10)
    ax_stack.grid(True, alpha=0.2, axis="x")

    # ─── Panel C: Vulnerability components by latitude ───
    ax_comp = fig.add_subplot(gs[1, 1])

    ax_comp.plot(eco_df["lat_center"], 1 - eco_df["mean_diversity"],
                 "b-o", markersize=3, linewidth=2, alpha=0.7,
                 label="Diversity deficit (1 - div_norm)")
    ax_comp.plot(eco_df["lat_center"], eco_df["mean_exposure"],
                 "r-s", markersize=3, linewidth=2, alpha=0.7,
                 label="Climate exposure")
    ax_comp.plot(eco_df["lat_center"], eco_df["mean_uncertainty"],
                 "gray", linestyle="--", linewidth=1.5, alpha=0.6,
                 label="Model uncertainty")
    ax_comp.plot(eco_df["lat_center"], eco_df["mean_vulnerability"],
                 "k-", linewidth=2.5, alpha=0.9,
                 label="Vulnerability score")

    ax_comp.set_xlabel("Latitude", fontsize=12)
    ax_comp.set_ylabel("Score (0-1)", fontsize=12)
    ax_comp.set_title("Vulnerability Components Along Coast",
                      fontsize=14, fontweight="bold")
    ax_comp.legend(fontsize=10, loc="best", frameon=True)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.tick_params(labelsize=10)
    ax_comp.set_ylim([0, 1])

    fig.suptitle(
        f"Ecosystem-Level Climate Change Resilience — California Coast\n"
        f"{results_df['species'].nunique()} species, {len(results_df)} predictions",
        fontsize=20, fontweight="bold", y=0.98,
    )
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"ecosystem_resilience.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'ecosystem_resilience.png'}")

    # ── Ecosystem summary by region ──
    eco_df["region"] = pd.cut(
        eco_df["lat_center"],
        bins=[29, 34, 37, 43],
        labels=["Southern CA (29-34°N)", "Central CA (34-37°N)", "Northern CA (37-43°N)"],
    )
    print("\n  Ecosystem resilience by region:")
    for region, rdf in eco_df.groupby("region", observed=True):
        print(f"    {region}:")
        print(f"      Mean vulnerability: {rdf['mean_vulnerability'].mean():.3f}")
        print(f"      Mean species coverage: {rdf['n_species'].mean():.1f}")
        print(f"      % Critical: {rdf['pct_critical'].mean()*100:.1f}%")
        print(f"      % Resilient: {rdf['pct_resilient'].mean()*100:.1f}%")

    # ── Summary statistics ──
    print("\n" + "=" * 60)
    print("RESILIENCE ASSESSMENT SUMMARY")
    print("=" * 60)
    print(f"\nSpecies assessed: {results_df['species'].nunique()}")
    print(f"Total predictions: {len(results_df)}")

    for cls in ["Resilient", "At Risk", "Latent Vulnerability", "Critical"]:
        pct = (results_df["resilience_class"] == cls).mean() * 100
        print(f"  {cls}: {pct:.1f}%")

    print(f"\nMost vulnerable species (>50% Critical predictions):")
    critical_sp = sp_vuln[sp_vuln["pct_critical"] > 0.5]
    for _, row in critical_sp.iterrows():
        print(f"  {row['species']} ({row['metric_type']}): "
              f"vulnerability={row['mean_vulnerability']:.3f}, "
              f"critical={row['pct_critical']:.1%}")

    print(f"\nMost resilient species (<10% Critical):")
    resilient_sp = sp_vuln[sp_vuln["pct_critical"] < 0.1].head(5)
    for _, row in resilient_sp.iterrows():
        print(f"  {row['species']} ({row['metric_type']}): "
              f"vulnerability={row['mean_vulnerability']:.3f}")


if __name__ == "__main__":
    main()

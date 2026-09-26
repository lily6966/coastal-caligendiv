#!/usr/bin/env python3
"""
Step 4: CNP prediction along California coast with uncertainty visualization.

For each species:
  - Uses all observed populations as context set
  - Predicts diversity at a coastal grid with uncertainty (μ ± σ)
  - Generates 4 figure panels

Outputs:
  figures/coastal_transects.png       — Per-species predictions with uncertainty ribbons
  figures/obs_vs_pred.png             — Observed vs predicted scatter with error bars
  figures/calibration.png             — Uncertainty calibration plot
  figures/california_diversity_map.png — Map view of predicted diversity
  data/processed/predictions.csv      — All predictions with uncertainty
"""

import pickle, sys, warnings
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
import os
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG_DIR = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
import env_present
import land_use
import lulc_model
ConditionalNeuralProcess = _m.ConditionalNeuralProcess
FeatureProcessor = _m.FeatureProcessor
NUMERIC_FEATURES = _m.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _m.CATEGORICAL_FEATURES
REPR_DIM = _m.REPR_DIM

# Use the 1 km-climate + NLCD land-use model (models/cnp_california_lulc.pt).
USE_LULC = True
CALI_FILE = "california_env_lulc.csv" if USE_LULC else "california_env.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model_and_processor():
    proc_file = "feature_processor_lulc.pkl" if USE_LULC else "feature_processor.pkl"
    model_file = "cnp_california_lulc.pt" if USE_LULC else "cnp_california_finetuned.pt"
    with open(MODEL_DIR / proc_file, "rb") as f:
        processor = pickle.load(f)

    n_numeric = len(NUMERIC_FEATURES) + (len(lulc_model.LULC_FEATURES) if USE_LULC else 0)
    model = ConditionalNeuralProcess(
        n_numeric=n_numeric,
        cat_cardinalities=processor.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2,
        n_species=processor.n_species,
        repr_dim=REPR_DIM,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / model_file, weights_only=True, map_location=DEVICE)
    )
    model.eval()
    return model, processor


def prepare_context(species_df, processor):
    """Encode observed populations as CNP context."""
    feats = processor.transform(species_df)
    return {
        "numeric": torch.tensor(feats["numeric"]).to(DEVICE),
        "cats": {k: torch.tensor(v).to(DEVICE) for k, v in feats["categoricals"].items()},
        "metric": torch.tensor(feats["metric_type"]).to(DEVICE),
        "species": torch.tensor(feats["species_id"]).to(DEVICE),
        "y": torch.tensor(feats["target"]).to(DEVICE),
    }


def _forward_z(model, context, target_df, processor):
    """Run the CNP and return standardized (transformed-space) mu, sigma + metric."""
    tgt_feats = processor.transform(target_df)
    tgt_numeric = torch.tensor(tgt_feats["numeric"]).to(DEVICE)
    tgt_cats = {k: torch.tensor(v).to(DEVICE) for k, v in tgt_feats["categoricals"].items()}
    tgt_metric = torch.tensor(tgt_feats["metric_type"]).to(DEVICE)
    tgt_species = torch.tensor(tgt_feats["species_id"]).to(DEVICE)

    with torch.no_grad():
        ctx_repr = model.encode_context(
            context["numeric"], context["cats"], context["metric"],
            context["species"], context["y"]
        )
        tgt_features = model.feature_encoder(
            tgt_numeric, tgt_cats, tgt_metric, tgt_species
        )
        z_mu, z_sigma = model.decode(ctx_repr, tgt_features)

    mt_arr = target_df["metric_type"].fillna("He").astype(str).values
    return z_mu.cpu().numpy(), z_sigma.cpu().numpy(), mt_arr


def predict_at_locations(model, context, target_df, processor):
    """CNP prediction in raw units: median mu and a delta-method symmetric sigma."""
    z_mu, z_sigma, mt_arr = _forward_z(model, context, target_df, processor)
    mu = processor.inverse_target(z_mu, mt_arr)
    sigma = processor.inverse_sigma(z_sigma, mt_arr, z_mu)
    return mu, sigma


def predict_intervals(model, context, target_df, processor, ks=(1.0, 2.0)):
    """Raw-unit median mu plus exact per-metric credible bounds at each k in ks.
    Bounds are computed in transformed space then inverse-linked, so pi bounds
    are asymmetric and strictly positive (never cross zero)."""
    z_mu, z_sigma, mt_arr = _forward_z(model, context, target_df, processor)
    mu = processor.inverse_target(z_mu, mt_arr)
    sigma = processor.inverse_sigma(z_sigma, mt_arr, z_mu)
    bounds = {k: processor.inverse_bounds(z_mu, z_sigma, mt_arr, k) for k in ks}
    return mu, sigma, bounds


# Transect resolution. 500 points over ~1570 km of coastline is ~3.1 km spacing,
# finer than the Bio-ORACLE marine layers (~6 km) and the WorldClim rasters
# (~18.5 km), so the limit is now the source data rather than the transect.
COAST_N_POINTS = 500


def make_coastal_grid(n_points=COAST_N_POINTS):
    """Generate California coastal grid points."""
    lats = np.linspace(30.5, 42.0, n_points)
    lons = np.interp(
        lats,
        [30.5, 32.5, 33.0, 34.0, 34.5, 35.0, 36.5, 37.5, 38.5, 40.0, 42.0],
        [-116.1, -117.2, -117.3, -118.5, -120.5, -120.7, -122.0, -122.5, -123.0, -124.2, -124.4],
    )
    return lats, lons


# Sample the spatial covariates from their source layers at each prediction
# point, instead of copying them from the nearest observed population. Set to
# False to restore the old nearest-observation behaviour.
SAMPLE_ENV_FROM_SOURCE = True


def make_target_df(lats, lons, species, metric_type, cali_env_df):
    """Target DataFrame for prediction.

    Species-level attributes (taxonomy, life form, marker, biome) come from the
    species' own records; the spatial environmental covariates are sampled at the
    actual coordinates so transect resolution is not capped by record spacing.
    """
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    df = pd.DataFrame({
        "Latitude": lats, "Longitude": lons,
        "species": species, "metric_type": metric_type,
        "gen_div": 0.0,
    })

    # Nearest observed record supplies the non-spatial, species-level columns
    env_cols = [c for c in cali_env_df.columns
                if c not in ["species", "Latitude", "Longitude", "gen_div",
                             "metric_type", "source", "is_expert"]]
    ref = cali_env_df[["Latitude", "Longitude"] + env_cols].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(ref[["Latitude", "Longitude"]].values)
    _, nearest = tree.query(np.column_stack([lats, lons]))
    for col in env_cols:
        df[col] = ref[col].values[nearest]

    if SAMPLE_ENV_FROM_SOURCE:
        sampled = env_present.present_env(lats, lons, cali_env_df)
        for col in env_present.ENV_COLS:
            if col in df.columns and sampled[col].notna().any():
                df[col] = sampled[col].values

    if USE_LULC:
        lu = land_use.pressure_at(lats, lons)
        known = (~lu["outside_nlcd"].values) & np.isfinite(lu["lulc_pressure"].values)
        for col in lulc_model.LULC_FEATURES:
            if col == "lulc_known":
                df["lulc_known"] = known.astype(float)
            else:
                df[col] = np.nan_to_num(lu[col].values, nan=0.0)

    return df


def leave_one_out_eval(model, species_df, processor):
    """Leave-one-out prediction for calibration assessment."""
    mus, sigmas, ys = [], [], []
    n = len(species_df)
    if n < 3:
        return np.array([]), np.array([]), np.array([])

    for i in range(n):
        ctx_df = species_df.drop(species_df.index[i])
        tgt_df = species_df.iloc[[i]]
        context = prepare_context(ctx_df, processor)
        mu, sigma = predict_at_locations(model, context, tgt_df, processor)
        mus.append(mu[0])
        sigmas.append(sigma[0])
        ys.append(tgt_df["gen_div"].values[0])

    return np.array(mus), np.array(sigmas), np.array(ys)


def main():
    model, processor = load_model_and_processor()
    cali_df = pd.read_csv(PROC / CALI_FILE)
    cali_df = _m.filter_ca_species(cali_df)
    coast_lats, coast_lons = make_coastal_grid()

    species_info = cali_df.groupby("species").agg(
        metric_type=("metric_type", "first"),
        n_pops=("gen_div", "count"),
        mean_div=("gen_div", "mean"),
    ).reset_index().sort_values("n_pops", ascending=False)

    # ── Figure 1: Coastal transects with uncertainty ──
    # Restrict each species' transect to its observed latitude range ± buffer
    LAT_BUFFER_DEG = 2.0
    top_species = species_info.head(9)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Predicted Genetic Diversity Along California Coast (CNP with Uncertainty)",
                 fontsize=14, y=0.98)

    all_predictions = []

    for idx, (_, sp_row) in enumerate(top_species.iterrows()):
        ax = axes[idx // 3, idx % 3]
        sp = sp_row["species"]
        mt = sp_row["metric_type"]

        obs = cali_df[cali_df["species"] == sp]
        context = prepare_context(obs, processor)
        target_df = make_target_df(coast_lats, coast_lons, sp, mt, cali_df)
        # Exact per-metric bounds (asymmetric & positive for pi via the log link)
        mu, sigma, bounds = predict_intervals(model, context, target_df, processor, ks=(1.0, 2.0))
        lo68, hi68 = bounds[1.0]
        lo95, hi95 = bounds[2.0]

        # Latitudes within the buffer of an actual observation are "in range";
        # the rest are drawn in grey (kept, not removed) so it is clear where the
        # species is not observed rather than stretching the coloured line there.
        obs_lats = obs["Latitude"].to_numpy()
        dist_to_obs = np.abs(coast_lats[:, None] - obs_lats[None, :]).min(axis=1)
        in_range = dist_to_obs <= LAT_BUFFER_DEG

        # Out-of-range portion in grey (drawn first, underneath)
        ax.fill_between(coast_lats, lo95, hi95,
                        where=~in_range, interpolate=True, alpha=0.10, color="grey")
        ax.fill_between(coast_lats, lo68, hi68,
                        where=~in_range, interpolate=True, alpha=0.18, color="grey")
        ax.plot(coast_lats, mu, color="grey", linewidth=2, alpha=0.8,
                label="Out of range (>2° from obs)")
        # In-range portion in blue, overlaid on top
        ax.fill_between(coast_lats, lo95, hi95,
                        where=in_range, interpolate=True, alpha=0.15, color="blue", label="95% CI")
        ax.fill_between(coast_lats, lo68, hi68,
                        where=in_range, interpolate=True, alpha=0.3, color="blue", label="68% CI")
        ax.plot(coast_lats, np.where(in_range, mu, np.nan), "b-", linewidth=2, label="Predicted μ")
        ax.scatter(obs["Latitude"], obs["gen_div"], c="red", s=25,
                   alpha=0.8, zorder=5, label="Observed")
        ax.set_title(f"{sp}\n({mt}, n={len(obs)})", fontsize=9)
        ax.set_xlabel("Latitude")
        ax.set_ylabel(f"{'He' if mt == 'He' else 'π'}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        for i in range(len(coast_lats)):
            all_predictions.append({
                "species": sp, "metric_type": mt,
                "Latitude": coast_lats[i], "Longitude": coast_lons[i],
                "pred_mu": mu[i], "pred_sigma": sigma[i],
                "pred_lo95": lo95[i], "pred_hi95": hi95[i],
                "in_range": bool(in_range[i]),
            })

    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"coastal_transects.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'coastal_transects.pdf'}")

    # ── Figure 2: Leave-one-out performance ──
    # Panels (a) and (b) are raw observed vs predicted, as before. Panel (c) is
    # the diagnostic that matters: once each species is centred on its own mean,
    # how much of the WITHIN-species variation does the model actually track?
    # Overall R² is dominated by between-species differences and hides this.
    loo = []
    for mt_name in ["He", "pi"]:
        mt_sp = cali_df[cali_df["metric_type"] == mt_name]
        for sp in mt_sp["species"].unique():
            obs = mt_sp[mt_sp["species"] == sp]
            if len(obs) < 3:
                continue
            mus, sigmas, ys = leave_one_out_eval(model, obs, processor)
            if len(mus) == 0:
                continue
            for m_, s_, y_ in zip(mus, sigmas, ys):
                loo.append({"species": sp, "metric_type": mt_name,
                            "obs": y_, "pred": m_, "sigma": s_})
    loo = pd.DataFrame(loo)
    loo.to_csv(PROC / "loo_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, mt_name, title in zip(axes[:2], ["He", "pi"],
                                  ["(a) Heterozygosity species", "(b) Nucleotide diversity species"]):
        d = loo[loo["metric_type"] == mt_name]
        for sp in sorted(d["species"].unique()):
            g = d[d["species"] == sp]
            ax.errorbar(g["obs"], g["pred"], yerr=1.96 * g["sigma"], fmt="o", ms=4,
                        alpha=0.5, capsize=2, label=sp[:20])
        lim = [0, max(d["obs"].max(), d["pred"].max()) * 1.05]
        ax.plot(lim, lim, "k--", alpha=0.4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        r2 = 1 - ((d["obs"] - d["pred"]) ** 2).sum() / ((d["obs"] - d["obs"].mean()) ** 2).sum()
        ratio = d["pred"].std() / d["obs"].std()
        ax.set_xlabel(f"Observed {mt_name}")
        ax.set_ylabel(f"Predicted {mt_name}")
        ax.set_title(f"{title}\nR² = {r2:+.2f} (mostly between-species) | "
                     f"predicted spread = {100*ratio:.0f}% of observed", fontsize=10)
        ax.legend(fontsize=5, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)

    # (c) within-species anomalies, standardized per species so He and pi share an axis
    ax = axes[2]
    rows = []
    for (sp, mt_name), g in loo.groupby(["species", "metric_type"]):
        if len(g) < 5 or g["obs"].std() < 1e-12:
            continue
        rows.append(pd.DataFrame({
            "species": sp, "metric_type": mt_name,
            "obs_z": (g["obs"] - g["obs"].mean()) / g["obs"].std(),
            "pred_z": (g["pred"] - g["pred"].mean()) /
                      (g["pred"].std() if g["pred"].std() > 1e-12 else np.nan),
        }))
    z = pd.concat(rows) if rows else pd.DataFrame(columns=["obs_z", "pred_z"])
    z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=["obs_z", "pred_z"])
    for mt_name, mk in [("He", "o"), ("pi", "^")]:
        g = z[z["metric_type"] == mt_name]
        ax.scatter(g["obs_z"], g["pred_z"], s=18, alpha=0.5, marker=mk, label=mt_name)
    ax.plot([-3, 3], [-3, 3], "k--", alpha=0.4)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    r = np.corrcoef(z["obs_z"], z["pred_z"])[0, 1] if len(z) > 2 else np.nan
    per_sp = z.groupby("species").apply(
        lambda g: np.corrcoef(g["obs_z"], g["pred_z"])[0, 1] if len(g) > 2 else np.nan)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("Observed anomaly (SD from that species' mean)")
    ax.set_ylabel("Predicted anomaly (SD from that species' mean)")
    ax.set_title(f"(c) Within-species skill\npooled r = {r:+.2f} | "
                 f"median per-species r = {np.nanmedian(per_sp):+.2f} "
                 f"({int((per_sp < 0).sum())}/{len(per_sp)} negative)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Leave-One-Out Predictions (CNP) — overall fit vs within-species skill",
                 fontsize=13)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"obs_vs_pred.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'obs_vs_pred.pdf'}")

    # ── Figure 3: Calibration plot ──
    all_mus, all_sigmas, all_ys = [], [], []
    for sp in cali_df["species"].unique():
        obs = cali_df[cali_df["species"] == sp]
        if len(obs) < 3:
            continue
        mus, sigmas, ys = leave_one_out_eval(model, obs, processor)
        all_mus.extend(mus)
        all_sigmas.extend(sigmas)
        all_ys.extend(ys)

    all_mus = np.array(all_mus)
    all_sigmas = np.array(all_sigmas)
    all_ys = np.array(all_ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    nominal_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    empirical = []
    for level in nominal_levels:
        z = {0.1: 0.126, 0.2: 0.253, 0.3: 0.385, 0.4: 0.524, 0.5: 0.674,
             0.6: 0.842, 0.7: 1.036, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96}[level]
        lo = all_mus - z * all_sigmas
        hi = all_mus + z * all_sigmas
        cov = np.mean((all_ys >= lo) & (all_ys <= hi))
        empirical.append(cov)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(nominal_levels, empirical, "bo-", label="CNP")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Uncertainty Calibration (Leave-One-Out)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"calibration.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'calibration.pdf'}")

    # ── Figure 4: Map view ──
    fig, ax = plt.subplots(figsize=(8, 14))
    # Plot predictions for the species with most populations
    top_sp = species_info.iloc[0]["species"]
    top_mt = species_info.iloc[0]["metric_type"]
    obs = cali_df[cali_df["species"] == top_sp]
    context = prepare_context(obs, processor)
    target_df = make_target_df(coast_lats, coast_lons, top_sp, top_mt, cali_df)
    mu, sigma = predict_at_locations(model, context, target_df, processor)

    sc = ax.scatter(coast_lons, coast_lats, c=mu, cmap="viridis", s=40,
                    vmin=mu.min(), vmax=mu.max(), zorder=2)
    ax.scatter(obs["Longitude"], obs["Latitude"], c=obs["gen_div"],
               cmap="viridis", s=80, edgecolors="red", linewidth=1.5,
               vmin=mu.min(), vmax=mu.max(), zorder=3)
    plt.colorbar(sc, ax=ax, label=f"{'He' if top_mt == 'He' else 'π'}", shrink=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted Genetic Diversity: {top_sp}\n"
                 f"(Red-edged = observed, circles = predicted)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for _ext, _kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG_DIR / f"california_diversity_map.{_ext}", bbox_inches="tight", **_kw)
    plt.close()
    print(f"Saved: {FIG_DIR / 'california_diversity_map.pdf'}")

    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(PROC / "predictions.csv", index=False)
    print(f"Saved: {PROC / 'predictions.csv'} ({len(pred_df)} predictions)")


if __name__ == "__main__":
    main()

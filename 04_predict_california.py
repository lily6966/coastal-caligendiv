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
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
ConditionalNeuralProcess = _m.ConditionalNeuralProcess
FeatureProcessor = _m.FeatureProcessor
NUMERIC_FEATURES = _m.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _m.CATEGORICAL_FEATURES
REPR_DIM = _m.REPR_DIM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model_and_processor():
    with open(MODEL_DIR / "feature_processor.pkl", "rb") as f:
        processor = pickle.load(f)

    model = ConditionalNeuralProcess(
        n_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=processor.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2,
        n_species=processor.n_species,
        repr_dim=REPR_DIM,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "cnp_california_finetuned.pt",
                    weights_only=True, map_location=DEVICE)
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


def predict_at_locations(model, context, target_df, processor):
    """CNP prediction: encode context → decode at target locations."""
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
        mu, sigma = model.decode(ctx_repr, tgt_features)

    return mu.cpu().numpy(), sigma.cpu().numpy()


def make_coastal_grid(n_points=150):
    """Generate California coastal grid points."""
    lats = np.linspace(30.5, 42.0, n_points)
    lons = np.interp(
        lats,
        [30.5, 32.5, 33.0, 34.0, 34.5, 35.0, 36.5, 37.5, 38.5, 40.0, 42.0],
        [-116.1, -117.2, -117.3, -118.5, -120.5, -120.7, -122.0, -122.5, -123.0, -124.2, -124.4],
    )
    return lats, lons


def make_target_df(lats, lons, species, metric_type, cali_env_df):
    """Create a target DataFrame with environmental features from nearest known point."""
    df = pd.DataFrame({
        "Latitude": lats, "Longitude": lons,
        "species": species, "metric_type": metric_type,
        "gen_div": 0.0,
    })
    # Copy env columns from nearest California point
    env_cols = [c for c in cali_env_df.columns
                if c not in ["species", "Latitude", "Longitude", "gen_div",
                             "metric_type", "source", "is_expert"]]
    ref = cali_env_df[["Latitude", "Longitude"] + env_cols].drop_duplicates()
    for col in env_cols:
        df[col] = np.nan
    for i in range(len(df)):
        dists = np.abs(ref["Latitude"] - df.iloc[i]["Latitude"]) + \
                np.abs(ref["Longitude"] - df.iloc[i]["Longitude"])
        nearest = dists.idxmin()
        for col in env_cols:
            df.at[i, col] = ref.at[nearest, col]
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
    cali_df = pd.read_csv(PROC / "california_env.csv")
    coast_lats, coast_lons = make_coastal_grid()

    species_info = cali_df.groupby("species").agg(
        metric_type=("metric_type", "first"),
        n_pops=("gen_div", "count"),
        mean_div=("gen_div", "mean"),
    ).reset_index().sort_values("n_pops", ascending=False)

    # ── Figure 1: Coastal transects with uncertainty ──
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
        mu, sigma = predict_at_locations(model, context, target_df, processor)

        ax.fill_between(coast_lats, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.15, color="blue", label="95% CI")
        ax.fill_between(coast_lats, mu - sigma, mu + sigma,
                        alpha=0.3, color="blue", label="68% CI")
        ax.plot(coast_lats, mu, "b-", linewidth=2, label="Predicted μ")
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
            })

    plt.tight_layout()
    plt.savefig(FIG_DIR / "coastal_transects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'coastal_transects.png'}")

    # ── Figure 2: Observed vs Predicted with error bars ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mt_name, title in zip(axes, ["He", "pi"],
                                   ["Heterozygosity species", "Nucleotide diversity species"]):
        mt_sp = cali_df[cali_df["metric_type"] == mt_name]
        for sp in mt_sp["species"].unique():
            obs = mt_sp[mt_sp["species"] == sp]
            if len(obs) < 3:
                continue
            mus, sigmas, ys = leave_one_out_eval(model, obs, processor)
            if len(mus) == 0:
                continue
            ax.errorbar(ys, mus, yerr=1.96 * sigmas, fmt="o", ms=4, alpha=0.5,
                        capsize=2, label=sp[:20])
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k--", alpha=0.4)
        ax.set_xlabel(f"Observed {mt_name}")
        ax.set_ylabel(f"Predicted {mt_name}")
        ax.set_title(title)
        ax.legend(fontsize=5, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Leave-One-Out Predictions (CNP)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "obs_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'obs_vs_pred.png'}")

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
    plt.savefig(FIG_DIR / "calibration.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'calibration.png'}")

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
    plt.savefig(FIG_DIR / "california_diversity_map.png", dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR / 'california_diversity_map.png'}")

    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(PROC / "predictions.csv", index=False)
    print(f"Saved: {PROC / 'predictions.csv'} ({len(pred_df)} predictions)")


if __name__ == "__main__":
    main()

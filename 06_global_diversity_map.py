#!/usr/bin/env python3
"""
Step 6: Global genetic diversity map.

Uses the pre-trained CNP to predict genetic diversity across the globe,
using species with the most observed populations as context.

For each well-sampled species:
  - All observed populations serve as context
  - Predictions are made at the observed locations of OTHER populations
    within the same species (leave-one-out style) AND across a global grid

Outputs:
  figures/global_diversity_observed.png    — Map of all observed He values
  figures/global_diversity_predicted.png   — Per-species predicted diversity maps
  figures/global_diversity_grid.png        — Gridded mean predicted diversity
  figures/global_latitudinal_gradient.png  — Diversity vs latitude pattern
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
PROC = ROOT / "data" / "processed"
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


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
    feats = processor.transform(species_df)
    return {
        "numeric": torch.tensor(feats["numeric"]).to(DEVICE),
        "cats": {k: torch.tensor(v).to(DEVICE) for k, v in feats["categoricals"].items()},
        "metric": torch.tensor(feats["metric_type"]).to(DEVICE),
        "species": torch.tensor(feats["species_id"]).to(DEVICE),
        "y": torch.tensor(feats["target"]).to(DEVICE),
    }


def predict_at_locations(model, context, target_df, processor):
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


def make_global_target_df(lats, lons, species, metric_type, global_df,
                          _cache={}):
    """Create target DataFrame for global grid with env features from nearest point.

    Uses cKDTree for fast nearest-neighbor lookup. The tree is cached across calls.
    """
    from scipy.spatial import cKDTree

    skip_cols = {"species", "Latitude", "Longitude", "gen_div",
                 "metric_type", "source", "is_expert", "Spec_id",
                 "Study_id", "Pop_id", "N", "Ar", "Ho", "He",
                 "GD_Nei", "F_is", "Ploidy", "N_loci"}
    env_cols = [c for c in global_df.columns if c not in skip_cols]

    # Cache reference data and KDTree
    cache_key = id(global_df)
    if cache_key not in _cache:
        ref = global_df[["Latitude", "Longitude"] + env_cols].drop_duplicates().reset_index(drop=True)
        tree = cKDTree(ref[["Latitude", "Longitude"]].values)
        _cache[cache_key] = (ref, tree, env_cols)
    ref, tree, env_cols = _cache[cache_key]

    # Vectorized nearest-neighbor lookup
    query_coords = np.column_stack([lats, lons])
    _, nearest_idx = tree.query(query_coords)

    # Build DataFrame from nearest reference rows
    nearest_rows = ref.iloc[nearest_idx][env_cols].reset_index(drop=True)
    df = pd.DataFrame({
        "Latitude": lats,
        "Longitude": lons,
        "species": species,
        "metric_type": metric_type,
        "gen_div": 0.0,
    })
    for col in env_cols:
        df[col] = nearest_rows[col].values

    return df


def main():
    model, processor = load_model_and_processor()
    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)

    print(f"Global: {len(global_df)} records, {global_df['species'].nunique()} species")

    # ══════════════════════════════════════════════════
    # Figure 1: Observed global genetic diversity
    # ══════════════════════════════════════════════════
    print("\nPlotting observed global diversity...")
    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": None})

    sc = ax.scatter(
        global_df["Longitude"], global_df["Latitude"],
        c=global_df["gen_div"], cmap="viridis", s=3, alpha=0.4,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="He (observed)", shrink=0.6, pad=0.02)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Observed Genetic Diversity (He) — {len(global_df):,} populations, "
                 f"{global_df['species'].nunique()} species")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.grid(True, alpha=0.2)

    # Add continent outlines approximation
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "global_diversity_observed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'global_diversity_observed.png'}")

    # ══════════════════════════════════════════════════
    # Figure 2: Per-species predicted diversity maps
    # ══════════════════════════════════════════════════
    # Pick 9 well-sampled species with broad geographic ranges
    sp_stats = global_df.groupby("species").agg(
        n_pops=("gen_div", "count"),
        mean_div=("gen_div", "mean"),
        lat_range=("Latitude", lambda x: x.max() - x.min()),
        lon_range=("Longitude", lambda x: x.max() - x.min()),
        lat_min=("Latitude", "min"),
        lat_max=("Latitude", "max"),
    ).reset_index()
    # Require at least 20 pops and reasonable geographic spread
    well_sampled = sp_stats[(sp_stats["n_pops"] >= 20) & (sp_stats["lat_range"] > 5)]
    well_sampled = well_sampled.sort_values("n_pops", ascending=False)
    top9 = well_sampled.head(9)

    print(f"\nGenerating per-species predictions for {len(top9)} species...")

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    for idx, (_, sp_row) in enumerate(top9.iterrows()):
        ax = axes[idx // 3, idx % 3]
        sp = sp_row["species"]
        obs = global_df[global_df["species"] == sp]
        n_obs = len(obs)

        # Use 70% as context, predict at remaining 30%
        np.random.seed(42)
        n_ctx = max(3, int(n_obs * 0.7))
        perm = np.random.permutation(n_obs)
        ctx_idx = perm[:n_ctx]
        tgt_idx = perm[n_ctx:]

        ctx_df = obs.iloc[ctx_idx]
        tgt_df = obs.iloc[tgt_idx].copy()

        context = prepare_context(ctx_df, processor)
        mu, sigma = predict_at_locations(model, context, tgt_df, processor)

        # Also predict at a grid within the species' range
        lat_buffer = max(2.0, (sp_row["lat_max"] - sp_row["lat_min"]) * 0.15)
        grid_lats = np.linspace(
            max(-85, sp_row["lat_min"] - lat_buffer),
            min(85, sp_row["lat_max"] + lat_buffer),
            50
        )
        obs_lon_min = obs["Longitude"].min()
        obs_lon_max = obs["Longitude"].max()
        lon_buffer = max(2.0, (obs_lon_max - obs_lon_min) * 0.15)
        grid_lons = np.linspace(
            max(-180, obs_lon_min - lon_buffer),
            min(180, obs_lon_max + lon_buffer),
            50
        )
        grid_lat_mesh, grid_lon_mesh = np.meshgrid(grid_lats, grid_lons)
        flat_lats = grid_lat_mesh.ravel()
        flat_lons = grid_lon_mesh.ravel()

        grid_target = make_global_target_df(flat_lats, flat_lons, sp, "He", global_df)
        grid_mu, grid_sigma = predict_at_locations(model, context, grid_target, processor)

        # Plot gridded predictions as background
        sc = ax.scatter(
            flat_lons, flat_lats,
            c=grid_mu, cmap="viridis", s=8, alpha=0.3,
            vmin=0, vmax=1, marker="s",
        )

        # Overlay observed points
        ax.scatter(
            obs["Longitude"], obs["Latitude"],
            c=obs["gen_div"], cmap="viridis", s=30, alpha=0.8,
            edgecolors="black", linewidth=0.5,
            vmin=0, vmax=1, zorder=3,
        )

        # Overlay target predictions with error indication
        ax.scatter(
            tgt_df["Longitude"], tgt_df["Latitude"],
            c=mu, cmap="viridis", s=50, alpha=0.9,
            edgecolors="red", linewidth=1.2,
            vmin=0, vmax=1, zorder=4,
        )

        # RMSE for this species
        rmse = np.sqrt(np.mean((mu - tgt_df["gen_div"].values)**2))

        ax.set_title(f"{sp}\n(n={n_obs}, RMSE={rmse:.3f})", fontsize=9)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(True, alpha=0.2)

        if idx == 0:
            plt.colorbar(sc, ax=ax, label="He", shrink=0.7)

    fig.suptitle("Per-Species Predicted Genetic Diversity (CNP)\n"
                 "Background = grid predictions, Black-edged = observed, "
                 "Red-edged = held-out predictions",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "global_diversity_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'global_diversity_predicted.png'}")

    # ══════════════════════════════════════════════════
    # Figure 3: Gridded mean predicted diversity
    # ══════════════════════════════════════════════════
    print("\nGenerating gridded global diversity predictions...")

    # Select ~30 well-sampled species across diverse taxa
    diverse_sp = well_sampled.head(30)

    # Collect predictions at observed locations for all selected species
    all_pred_lats = []
    all_pred_lons = []
    all_pred_mu = []
    all_obs_div = []

    for _, sp_row in diverse_sp.iterrows():
        sp = sp_row["species"]
        obs = global_df[global_df["species"] == sp]

        if len(obs) < 5:
            continue

        # Leave-one-out predictions
        for i in range(len(obs)):
            ctx_df = obs.drop(obs.index[i])
            tgt_df = obs.iloc[[i]].copy()
            context = prepare_context(ctx_df, processor)
            mu, sigma = predict_at_locations(model, context, tgt_df, processor)
            all_pred_lats.append(tgt_df["Latitude"].values[0])
            all_pred_lons.append(tgt_df["Longitude"].values[0])
            all_pred_mu.append(mu[0])
            all_obs_div.append(tgt_df["gen_div"].values[0])

    all_pred_lats = np.array(all_pred_lats)
    all_pred_lons = np.array(all_pred_lons)
    all_pred_mu = np.array(all_pred_mu)
    all_obs_div = np.array(all_obs_div)

    print(f"  Generated {len(all_pred_mu)} leave-one-out predictions "
          f"across {len(diverse_sp)} species")

    # Bin into grid cells for a smoothed map
    lat_bins = np.arange(-80, 81, 5)
    lon_bins = np.arange(-180, 181, 5)

    # Observed gridded
    obs_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    pred_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    count_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lat_bins)-1):
        for j in range(len(lon_bins)-1):
            mask = (
                (global_df["Latitude"] >= lat_bins[i]) &
                (global_df["Latitude"] < lat_bins[i+1]) &
                (global_df["Longitude"] >= lon_bins[j]) &
                (global_df["Longitude"] < lon_bins[j+1])
            )
            if mask.sum() > 0:
                obs_grid[i, j] = global_df.loc[mask, "gen_div"].mean()
                count_grid[i, j] = mask.sum()

            pred_mask = (
                (all_pred_lats >= lat_bins[i]) &
                (all_pred_lats < lat_bins[i+1]) &
                (all_pred_lons >= lon_bins[j]) &
                (all_pred_lons < lon_bins[j+1])
            )
            if pred_mask.sum() > 0:
                pred_grid[i, j] = all_pred_mu[pred_mask].mean()

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel A: Observed gridded diversity
    ax = axes[0]
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    im = ax.pcolormesh(lon_bins, lat_bins, obs_grid, cmap="viridis",
                        vmin=0, vmax=1, shading="flat")
    plt.colorbar(im, ax=ax, label="Mean He", shrink=0.6)
    ax.set_title(f"Observed Mean Genetic Diversity (5x5 deg grid)\n"
                 f"{len(global_df):,} populations", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-80, 80])
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)

    # Panel B: Predicted gridded diversity (LOO)
    ax = axes[1]
    im = ax.pcolormesh(lon_bins, lat_bins, pred_grid, cmap="viridis",
                        vmin=0, vmax=1, shading="flat")
    plt.colorbar(im, ax=ax, label="Mean predicted He", shrink=0.6)
    ax.set_title(f"CNP Leave-One-Out Predicted Diversity (5x5 deg grid)\n"
                 f"{len(all_pred_mu)} predictions across {len(diverse_sp)} species",
                 fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-80, 80])
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "global_diversity_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'global_diversity_grid.png'}")

    # ══════════════════════════════════════════════════
    # Figure 4: Latitudinal diversity gradient
    # ══════════════════════════════════════════════════
    print("\nPlotting latitudinal gradient...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Observed
    ax = axes[0]
    ax.scatter(global_df["abs_latitude"], global_df["gen_div"],
               s=2, alpha=0.1, color="steelblue")
    # Binned mean
    lat_edges = np.arange(0, 81, 5)
    lat_mids = (lat_edges[:-1] + lat_edges[1:]) / 2
    binned_means = []
    binned_stds = []
    for i in range(len(lat_edges) - 1):
        mask = (global_df["abs_latitude"] >= lat_edges[i]) & \
               (global_df["abs_latitude"] < lat_edges[i+1])
        if mask.sum() > 5:
            binned_means.append(global_df.loc[mask, "gen_div"].mean())
            binned_stds.append(global_df.loc[mask, "gen_div"].std())
        else:
            binned_means.append(np.nan)
            binned_stds.append(np.nan)
    binned_means = np.array(binned_means)
    binned_stds = np.array(binned_stds)
    valid = ~np.isnan(binned_means)
    ax.plot(lat_mids[valid], binned_means[valid], "r-o", linewidth=2,
            markersize=5, label="Binned mean")
    ax.fill_between(lat_mids[valid],
                     binned_means[valid] - binned_stds[valid],
                     binned_means[valid] + binned_stds[valid],
                     alpha=0.2, color="red")
    ax.set_xlabel("Absolute Latitude")
    ax.set_ylabel("Genetic Diversity (He)")
    ax.set_title("Observed Latitudinal Diversity Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Predicted LOO
    ax = axes[1]
    abs_pred_lats = np.abs(all_pred_lats)
    ax.scatter(abs_pred_lats, all_pred_mu, s=5, alpha=0.3, color="steelblue",
               label="LOO predicted")
    ax.scatter(abs_pred_lats, all_obs_div, s=5, alpha=0.3, color="coral",
               label="Observed")

    # Binned means for predicted
    pred_binned = []
    obs_binned = []
    for i in range(len(lat_edges) - 1):
        mask = (abs_pred_lats >= lat_edges[i]) & (abs_pred_lats < lat_edges[i+1])
        if mask.sum() > 3:
            pred_binned.append(all_pred_mu[mask].mean())
            obs_binned.append(all_obs_div[mask].mean())
        else:
            pred_binned.append(np.nan)
            obs_binned.append(np.nan)
    pred_binned = np.array(pred_binned)
    obs_binned = np.array(obs_binned)
    valid = ~np.isnan(pred_binned)
    ax.plot(lat_mids[valid], pred_binned[valid], "b-o", linewidth=2,
            markersize=5, label="Predicted mean")
    ax.plot(lat_mids[valid], obs_binned[valid], "r--s", linewidth=2,
            markersize=5, label="Observed mean")
    ax.set_xlabel("Absolute Latitude")
    ax.set_ylabel("Genetic Diversity (He)")
    ax.set_title("CNP Leave-One-Out: Predicted vs Observed Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Global LOO RMSE
    loo_rmse = np.sqrt(np.mean((all_pred_mu - all_obs_div)**2))
    ss_res = np.sum((all_obs_div - all_pred_mu)**2)
    ss_tot = np.sum((all_obs_div - all_obs_div.mean())**2)
    loo_r2 = 1 - ss_res / (ss_tot + 1e-8)
    fig.suptitle(f"Latitudinal Diversity Gradient — Global LOO: "
                 f"RMSE={loo_rmse:.4f}, R²={loo_r2:.4f}",
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "global_latitudinal_gradient.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'global_latitudinal_gradient.png'}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("GLOBAL DIVERSITY MAP SUMMARY")
    print("=" * 60)
    print(f"Total observations: {len(global_df):,}")
    print(f"Species mapped: {len(diverse_sp)}")
    print(f"LOO predictions: {len(all_pred_mu)}")
    print(f"LOO RMSE: {loo_rmse:.4f}")
    print(f"LOO R²: {loo_r2:.4f}")

    # Sampling coverage
    filled_cells = np.sum(~np.isnan(obs_grid))
    total_cells = obs_grid.size
    print(f"\nGrid coverage: {filled_cells}/{total_cells} cells "
          f"({filled_cells/total_cells*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()

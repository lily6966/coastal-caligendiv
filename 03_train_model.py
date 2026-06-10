#!/usr/bin/env python3
"""
Step 3: Conditional Neural Process for genetic diversity prediction.

Architecture:
  - Feature encoder: numeric covariates + categorical embeddings + species embedding
  - FiLM conditioning on metric type (He vs π) — single model for both scales
  - Observation encoder: (features, gen_div) → latent repr
  - Mean-pooling aggregator (permutation invariant)
  - Decoder: (aggregated context, target features) → (μ, σ²)

Training:
  Phase 1: Pre-train on GenDivRange (global, He-only)
    - Within-species episodic training
    - Cross-species episodes (pool by taxonomic order)
    - Pseudo-π episodes (low-He species treated as π to warm-start FiLM)
  Phase 2: Fine-tune on California (He + π)
    - Mixed global+local context augmentation
    - Lower learning rates, dropout regularization
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────
# Feature configuration
# ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Geographic
    "Latitude", "Longitude", "abs_latitude",
    # WorldClim bioclimatic
    "bio1", "bio4", "bio5", "bio6", "bio7",
    "bio12", "bio13", "bio14", "bio15",
    # Climate exposure proxies
    "temp_seasonality_norm", "precip_extremity", "climate_stress",
    # Marine (Bio-ORACLE)
    "sst_mean", "sst_max", "sst_range",
    "salinity_mean", "chl_mean", "o2_mean", "ph_mean",
    "is_marine",
]

CATEGORICAL_FEATURES = {
    "Life_form": 16,
    "Phylum_GBIF": 8,
    "Class_GBIF": 10,
    "Order_GBIF": 12,
    "Family_GBIF": 12,
    "Marker_type": 6,
    "BIOME": 10,
}

METRIC_EMBED_DIM = 8
SPECIES_EMBED_DIM = 16
REPR_DIM = 128


# ─────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────
class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.metric_encoder = LabelEncoder()
        self.species_encoder = LabelEncoder()
        self.cat_cardinalities = {}

    def fit_transform(self, df, fit=True):
        return self._process(df, fit=fit)

    def transform(self, df):
        return self._process(df, fit=False)

    def _process(self, df, fit=True):
        result = {}
        df = df.copy()

        # Numeric — fill missing with 0 (StandardScaler will center it)
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        num_data = df[NUMERIC_FEATURES].fillna(0.0)
        if fit:
            self.scaler.fit(num_data)
        result["numeric"] = self.scaler.transform(num_data).astype(np.float32)

        # Categorical
        result["categoricals"] = {}
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
            vals = df[col].fillna("unknown").astype(str)
            if fit:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(list(vals.unique()) + ["unknown"])
                self.cat_cardinalities[col] = len(self.label_encoders[col].classes_)
            known = set(self.label_encoders[col].classes_)
            vals = vals.apply(lambda x: x if x in known else "unknown")
            result["categoricals"][col] = (
                self.label_encoders[col].transform(vals) + 1
            ).astype(np.int64)

        # Metric type
        if fit:
            self.metric_encoder.fit(["He", "pi"])
        mt = df["metric_type"].fillna("He")
        result["metric_type"] = self.metric_encoder.transform(mt).astype(np.int64)

        # Species
        species = df["species"].fillna("unknown").astype(str)
        if fit:
            self.species_encoder.fit(list(species.unique()) + ["unknown"])
        known_sp = set(self.species_encoder.classes_)
        species = species.apply(lambda x: x if x in known_sp else "unknown")
        result["species_id"] = (
            self.species_encoder.transform(species) + 1
        ).astype(np.int64)

        result["target"] = df["gen_div"].values.astype(np.float32)
        return result

    @property
    def n_species(self):
        return len(self.species_encoder.classes_)


# ─────────────────────────────────────────────────────────
# CNP Datasets — episodic training
# ─────────────────────────────────────────────────────────
class WithinSpeciesCNPDataset:
    """Within-species episodes: random context/target split per species."""

    def __init__(self, df, processor, min_pops=3, pseudo_pi_frac=0.0):
        self.pseudo_pi_frac = pseudo_pi_frac
        self.groups = []
        self.pi_idx = int(processor.metric_encoder.transform(["pi"])[0])

        for sp, gdf in df.groupby("species"):
            if len(gdf) < min_pops + 1:
                continue
            feats = processor.transform(gdf)
            self.groups.append({
                "numeric": feats["numeric"],
                "categoricals": feats["categoricals"],
                "metric_type": feats["metric_type"],
                "species_id": feats["species_id"],
                "target": feats["target"],
                "n": len(gdf),
                "mean_div": feats["target"].mean(),
            })

        print(f"  WithinSpecies: {len(self.groups)} species groups")

    def sample_episode(self):
        group = self.groups[np.random.randint(len(self.groups))]
        n = group["n"]
        n_ctx = np.random.randint(2, max(3, int(n * 0.7)))
        perm = np.random.permutation(n)
        ctx_idx = perm[:n_ctx]
        tgt_idx = perm[n_ctx:] if n_ctx < n else perm[-1:]

        metric = group["metric_type"].copy()
        # Pseudo-π: for low-diversity species, sometimes flip to pi metric
        if self.pseudo_pi_frac > 0 and group["mean_div"] < 0.15:
            if np.random.random() < self.pseudo_pi_frac:
                metric = np.full_like(metric, self.pi_idx)

        return {
            "ctx_numeric": group["numeric"][ctx_idx],
            "ctx_cats": {k: v[ctx_idx] for k, v in group["categoricals"].items()},
            "ctx_metric": metric[ctx_idx],
            "ctx_species": group["species_id"][ctx_idx],
            "ctx_y": group["target"][ctx_idx],
            "tgt_numeric": group["numeric"][tgt_idx],
            "tgt_cats": {k: v[tgt_idx] for k, v in group["categoricals"].items()},
            "tgt_metric": metric[tgt_idx],
            "tgt_species": group["species_id"][tgt_idx],
            "tgt_y": group["target"][tgt_idx],
        }


class CrossSpeciesCNPDataset:
    """Cross-species episodes: pool populations within taxonomic order."""

    def __init__(self, df, processor, min_pops=6):
        self.groups = []
        group_col = "Order_GBIF" if "Order_GBIF" in df.columns else "Class_GBIF"

        for order, gdf in df.groupby(group_col):
            if pd.isna(order) or order == "unknown" or len(gdf) < min_pops:
                continue
            feats = processor.transform(gdf)
            self.groups.append({
                "numeric": feats["numeric"],
                "categoricals": feats["categoricals"],
                "metric_type": feats["metric_type"],
                "species_id": feats["species_id"],
                "target": feats["target"],
                "n": len(gdf),
            })

        print(f"  CrossSpecies: {len(self.groups)} taxonomic groups")

    def sample_episode(self):
        group = self.groups[np.random.randint(len(self.groups))]
        n = group["n"]
        n_ctx = np.random.randint(3, max(4, int(n * 0.5)))
        n_ctx = min(n_ctx, n - 1)
        perm = np.random.permutation(n)
        ctx_idx = perm[:n_ctx]
        tgt_idx = perm[n_ctx:]
        if len(tgt_idx) == 0:
            tgt_idx = perm[-1:]

        return {
            "ctx_numeric": group["numeric"][ctx_idx],
            "ctx_cats": {k: v[ctx_idx] for k, v in group["categoricals"].items()},
            "ctx_metric": group["metric_type"][ctx_idx],
            "ctx_species": group["species_id"][ctx_idx],
            "ctx_y": group["target"][ctx_idx],
            "tgt_numeric": group["numeric"][tgt_idx],
            "tgt_cats": {k: v[tgt_idx] for k, v in group["categoricals"].items()},
            "tgt_metric": group["metric_type"][tgt_idx],
            "tgt_species": group["species_id"][tgt_idx],
            "tgt_y": group["target"][tgt_idx],
        }


class MixedContextCNPDataset:
    """For Phase 2: California episodes augmented with global context from same family."""

    def __init__(self, cali_df, global_df, processor, min_pops=2):
        self.groups = []
        self.global_by_family = defaultdict(list)

        # Index global data by family
        if "Family_GBIF" in global_df.columns:
            for fam, gdf in global_df.groupby("Family_GBIF"):
                if pd.isna(fam) or fam == "unknown":
                    continue
                feats = processor.transform(gdf)
                self.global_by_family[fam] = {
                    "numeric": feats["numeric"],
                    "categoricals": feats["categoricals"],
                    "metric_type": feats["metric_type"],
                    "species_id": feats["species_id"],
                    "target": feats["target"],
                    "n": len(gdf),
                }

        for sp, gdf in cali_df.groupby("species"):
            if len(gdf) < min_pops + 1:
                continue
            feats = processor.transform(gdf)
            family = gdf["Family_GBIF"].iloc[0] if "Family_GBIF" in gdf.columns else "unknown"
            self.groups.append({
                "numeric": feats["numeric"],
                "categoricals": feats["categoricals"],
                "metric_type": feats["metric_type"],
                "species_id": feats["species_id"],
                "target": feats["target"],
                "n": len(gdf),
                "family": family,
            })

        print(f"  MixedContext: {len(self.groups)} CA species, "
              f"{len(self.global_by_family)} global families available")

    def sample_episode(self):
        group = self.groups[np.random.randint(len(self.groups))]
        n = group["n"]
        n_ctx = np.random.randint(2, max(3, int(n * 0.6)))
        perm = np.random.permutation(n)
        ctx_idx = perm[:n_ctx]
        tgt_idx = perm[n_ctx:] if n_ctx < n else perm[-1:]

        # Augment context with global records from same family
        ctx_numeric = [group["numeric"][ctx_idx]]
        ctx_cats = {k: [v[ctx_idx]] for k, v in group["categoricals"].items()}
        ctx_metric = [group["metric_type"][ctx_idx]]
        ctx_species = [group["species_id"][ctx_idx]]
        ctx_y = [group["target"][ctx_idx]]

        family = group["family"]
        if family in self.global_by_family and np.random.random() < 0.7:
            gdata = self.global_by_family[family]
            n_aug = min(5, gdata["n"])
            aug_idx = np.random.choice(gdata["n"], n_aug, replace=False)
            ctx_numeric.append(gdata["numeric"][aug_idx])
            for k in ctx_cats:
                ctx_cats[k].append(gdata["categoricals"][k][aug_idx])
            ctx_metric.append(gdata["metric_type"][aug_idx])
            ctx_species.append(gdata["species_id"][aug_idx])
            ctx_y.append(gdata["target"][aug_idx])

        return {
            "ctx_numeric": np.concatenate(ctx_numeric),
            "ctx_cats": {k: np.concatenate(v) for k, v in ctx_cats.items()},
            "ctx_metric": np.concatenate(ctx_metric),
            "ctx_species": np.concatenate(ctx_species),
            "ctx_y": np.concatenate(ctx_y),
            "tgt_numeric": group["numeric"][tgt_idx],
            "tgt_cats": {k: v[tgt_idx] for k, v in group["categoricals"].items()},
            "tgt_metric": group["metric_type"][tgt_idx],
            "tgt_species": group["species_id"][tgt_idx],
            "tgt_y": group["target"][tgt_idx],
        }


# ─────────────────────────────────────────────────────────
# Conditional Neural Process Model
# ─────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, cat_embed_dims,
                 n_metric_types, n_species, out_dim=REPR_DIM):
        super().__init__()
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(card + 1, dim, padding_idx=0)
            for name, (card, dim) in zip(
                cat_cardinalities.keys(),
                zip(cat_cardinalities.values(), cat_embed_dims.values())
            )
        })
        total_cat_dim = sum(cat_embed_dims.values())
        self.metric_embed = nn.Embedding(n_metric_types, METRIC_EMBED_DIM)
        self.species_embed = nn.Embedding(n_species + 1, SPECIES_EMBED_DIM, padding_idx=0)

        input_dim = n_numeric + total_cat_dim + SPECIES_EMBED_DIM
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )
        self.film_gamma = nn.Linear(METRIC_EMBED_DIM, out_dim)
        self.film_beta = nn.Linear(METRIC_EMBED_DIM, out_dim)
        self.out_dim = out_dim

    def forward(self, numeric, categoricals, metric_type, species_id):
        cat_embeds = [self.cat_embeddings[name](categoricals[name])
                      for name in self.cat_embeddings]
        cat_concat = torch.cat(cat_embeds, dim=-1)
        sp_embed = self.species_embed(species_id)
        x = torch.cat([numeric, cat_concat, sp_embed], dim=-1)
        h = self.mlp(x)
        m = self.metric_embed(metric_type)
        gamma = self.film_gamma(m)
        beta = self.film_beta(m)
        return gamma * h + beta


class ConditionalNeuralProcess(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, cat_embed_dims,
                 n_metric_types, n_species, repr_dim=REPR_DIM, dropout=0.0):
        super().__init__()

        self.feature_encoder = FeatureEncoder(
            n_numeric, cat_cardinalities, cat_embed_dims,
            n_metric_types, n_species, out_dim=repr_dim,
        )

        self.obs_encoder = nn.Sequential(
            nn.Linear(repr_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, repr_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(repr_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, 1)
        self.log_sigma_head = nn.Linear(128, 1)
        self.repr_dim = repr_dim

    def encode_context(self, ctx_numeric, ctx_cats, ctx_metric, ctx_species, ctx_y):
        features = self.feature_encoder(ctx_numeric, ctx_cats, ctx_metric, ctx_species)
        obs_input = torch.cat([features, ctx_y.unsqueeze(-1)], dim=-1)
        representations = self.obs_encoder(obs_input)
        return representations.mean(dim=0, keepdim=True)

    def decode(self, context_repr, tgt_features):
        context_expanded = context_repr.expand(tgt_features.shape[0], -1)
        combined = torch.cat([context_expanded, tgt_features], dim=-1)
        h = self.decoder(combined)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = torch.clamp(self.log_sigma_head(h).squeeze(-1), -10, 2)
        return mu, torch.exp(log_sigma)

    def forward_episode(self, ctx_numeric, ctx_cats, ctx_metric, ctx_species, ctx_y,
                        tgt_numeric, tgt_cats, tgt_metric, tgt_species):
        context_repr = self.encode_context(ctx_numeric, ctx_cats, ctx_metric, ctx_species, ctx_y)
        tgt_features = self.feature_encoder(tgt_numeric, tgt_cats, tgt_metric, tgt_species)
        return self.decode(context_repr, tgt_features)


# ─────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────
def to_device(arr):
    return torch.tensor(arr).to(DEVICE)


def cnp_loss(mu, sigma, target):
    dist = torch.distributions.Normal(mu, sigma + 1e-6)
    return -dist.log_prob(target).mean()


def run_episode(model, episode):
    ctx_numeric = to_device(episode["ctx_numeric"])
    ctx_cats = {k: to_device(v) for k, v in episode["ctx_cats"].items()}
    ctx_metric = to_device(episode["ctx_metric"])
    ctx_species = to_device(episode["ctx_species"])
    ctx_y = to_device(episode["ctx_y"])
    tgt_numeric = to_device(episode["tgt_numeric"])
    tgt_cats = {k: to_device(v) for k, v in episode["tgt_cats"].items()}
    tgt_metric = to_device(episode["tgt_metric"])
    tgt_species = to_device(episode["tgt_species"])
    tgt_y = to_device(episode["tgt_y"])

    mu, sigma = model.forward_episode(
        ctx_numeric, ctx_cats, ctx_metric, ctx_species, ctx_y,
        tgt_numeric, tgt_cats, tgt_metric, tgt_species,
    )
    return mu, sigma, tgt_y


def train_epoch(model, datasets, optimizer, n_episodes=300, mix_weights=None):
    """Train one epoch sampling from multiple datasets with given weights."""
    model.train()
    if mix_weights is None:
        mix_weights = [1.0 / len(datasets)] * len(datasets)

    total_loss = 0
    for _ in range(n_episodes):
        ds_idx = np.random.choice(len(datasets), p=mix_weights)
        episode = datasets[ds_idx].sample_episode()

        optimizer.zero_grad()
        mu, sigma, tgt_y = run_episode(model, episode)
        loss = cnp_loss(mu, sigma, tgt_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_episodes


def evaluate(model, dataset, n_episodes=100):
    model.eval()
    all_mu, all_sigma, all_y = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            episode = dataset.sample_episode()
            mu, sigma, tgt_y = run_episode(model, episode)
            all_mu.extend(mu.cpu().numpy())
            all_sigma.extend(sigma.cpu().numpy())
            all_y.extend(tgt_y.cpu().numpy())

    mu = np.array(all_mu)
    sigma = np.array(all_sigma)
    y = np.array(all_y)

    rmse = np.sqrt(np.mean((mu - y) ** 2))
    ss_res = np.sum((y - mu) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    ci_lo = mu - 1.96 * sigma
    ci_hi = mu + 1.96 * sigma
    coverage = np.mean((y >= ci_lo) & (y <= ci_hi))

    return {"rmse": rmse, "r2": r2, "mean_sigma": np.mean(sigma), "coverage_95": coverage}


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    global_path = PROC / "global_train_env.csv"
    cali_path = PROC / "california_env.csv"
    if not global_path.exists():
        print("Run 02_add_env_covariates.py first!")
        return

    global_df = pd.read_csv(global_path)
    cali_df = pd.read_csv(cali_path)

    # Ensure categorical columns exist
    for col in CATEGORICAL_FEATURES:
        if col not in cali_df.columns:
            cali_df[col] = "unknown"
        if col not in global_df.columns:
            global_df[col] = "unknown"

    print(f"Global: {len(global_df)} records, {global_df['species'].nunique()} species")
    print(f"California: {len(cali_df)} records, {cali_df['species'].nunique()} species")

    # Fit processor on combined data
    combined = pd.concat([global_df, cali_df], ignore_index=True)
    processor = FeatureProcessor()
    processor.fit_transform(combined, fit=True)

    # ════════════════════════════════════════════════
    # PHASE 1: Global pre-training
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Global pre-training")
    print("=" * 60)

    species_list = global_df["species"].unique()
    train_sp, val_sp = train_test_split(species_list, test_size=0.15, random_state=42)
    global_train = global_df[global_df["species"].isin(train_sp)]
    global_val = global_df[global_df["species"].isin(val_sp)]

    # Build datasets
    within_ds = WithinSpeciesCNPDataset(global_train, processor, min_pops=3, pseudo_pi_frac=0.2)
    cross_ds = CrossSpeciesCNPDataset(global_train, processor, min_pops=6)
    val_ds = WithinSpeciesCNPDataset(global_val, processor, min_pops=3)

    # Model
    model = ConditionalNeuralProcess(
        n_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=processor.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2,
        n_species=processor.n_species,
        repr_dim=REPR_DIM,
        dropout=0.0,
    ).to(DEVICE)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    P1_EPOCHS = 100
    P1_EPISODES = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=P1_EPOCHS)

    best_rmse = float("inf")
    patience_count = 0

    for epoch in range(1, P1_EPOCHS + 1):
        # 70% within-species, 30% cross-species
        loss = train_epoch(model, [within_ds, cross_ds], optimizer,
                           n_episodes=P1_EPISODES, mix_weights=[0.7, 0.3])
        metrics = evaluate(model, val_ds, n_episodes=100)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}  "
                  f"val_RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}  "
                  f"σ={metrics['mean_sigma']:.4f}  95%CI={metrics['coverage_95']:.3f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            patience_count = 0
            torch.save(model.state_dict(), MODEL_DIR / "cnp_global_pretrained.pt")
        else:
            patience_count += 1
            if patience_count >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "cnp_global_pretrained.pt", weights_only=True))
    metrics = evaluate(model, val_ds, n_episodes=200)
    print(f"\n  Best global: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, "
          f"σ={metrics['mean_sigma']:.4f}, coverage={metrics['coverage_95']:.3f}")

    # ════════════════════════════════════════════════
    # PHASE 2: California fine-tuning
    # ════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: California fine-tuning (He + π)")
    print("=" * 60)

    cali_species = cali_df["species"].unique()
    cali_train_sp, cali_test_sp = train_test_split(
        cali_species, test_size=0.3, random_state=42
    )
    cali_train = cali_df[cali_df["species"].isin(cali_train_sp)]
    cali_test = cali_df[cali_df["species"].isin(cali_test_sp)]

    # Zero-shot evaluation
    cali_test_ds = WithinSpeciesCNPDataset(cali_test, processor, min_pops=2)
    if len(cali_test_ds.groups) > 0:
        pre = evaluate(model, cali_test_ds, n_episodes=100)
        print(f"\n  Zero-shot: RMSE={pre['rmse']:.4f}, R²={pre['r2']:.4f}, "
              f"σ={pre['mean_sigma']:.4f}, coverage={pre['coverage_95']:.3f}")

    # Enable dropout for fine-tuning
    model_ft = ConditionalNeuralProcess(
        n_numeric=len(NUMERIC_FEATURES),
        cat_cardinalities=processor.cat_cardinalities,
        cat_embed_dims=CATEGORICAL_FEATURES,
        n_metric_types=2,
        n_species=processor.n_species,
        repr_dim=REPR_DIM,
        dropout=0.1,
    ).to(DEVICE)
    # Copy pre-trained weights
    pretrained_state = model.state_dict()
    ft_state = model_ft.state_dict()
    for k in pretrained_state:
        if k in ft_state and pretrained_state[k].shape == ft_state[k].shape:
            ft_state[k] = pretrained_state[k]
    model_ft.load_state_dict(ft_state)

    # Fine-tuning datasets
    within_cali = WithinSpeciesCNPDataset(cali_train, processor, min_pops=2)
    mixed_cali = MixedContextCNPDataset(cali_train, global_df, processor, min_pops=2)

    ft_optimizer = torch.optim.AdamW([
        {"params": model_ft.feature_encoder.mlp.parameters(), "lr": 2e-5},
        {"params": model_ft.feature_encoder.film_gamma.parameters(), "lr": 5e-4},
        {"params": model_ft.feature_encoder.film_beta.parameters(), "lr": 5e-4},
        {"params": model_ft.feature_encoder.metric_embed.parameters(), "lr": 5e-4},
        {"params": model_ft.feature_encoder.species_embed.parameters(), "lr": 2e-4},
        {"params": model_ft.feature_encoder.cat_embeddings.parameters(), "lr": 2e-5},
        {"params": model_ft.obs_encoder.parameters(), "lr": 1e-4},
        {"params": model_ft.decoder.parameters(), "lr": 2e-4},
        {"params": model_ft.mu_head.parameters(), "lr": 2e-4},
        {"params": model_ft.log_sigma_head.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)
    P2_EPOCHS = 80
    P2_EPISODES = 100
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=P2_EPOCHS)

    best_ft_rmse = float("inf")
    patience_count = 0

    for epoch in range(1, P2_EPOCHS + 1):
        # 50% within-species, 50% mixed global+local context
        datasets = [within_cali, mixed_cali]
        weights = [0.5, 0.5]
        # Fallback if one dataset is empty
        valid_ds = [(d, w) for d, w in zip(datasets, weights) if len(d.groups) > 0]
        if not valid_ds:
            print("  No valid training datasets!")
            break
        ds_list = [d for d, _ in valid_ds]
        w_list = [w for _, w in valid_ds]
        w_sum = sum(w_list)
        w_list = [w / w_sum for w in w_list]

        loss = train_epoch(model_ft, ds_list, ft_optimizer,
                           n_episodes=P2_EPISODES, mix_weights=w_list)

        if len(cali_test_ds.groups) > 0:
            metrics = evaluate(model_ft, cali_test_ds, n_episodes=50)
        else:
            metrics = {"rmse": 999, "r2": 0, "mean_sigma": 0, "coverage_95": 0}
        ft_scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}  "
                  f"test_RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}  "
                  f"σ={metrics['mean_sigma']:.4f}  coverage={metrics['coverage_95']:.3f}")

        if metrics["rmse"] < best_ft_rmse:
            best_ft_rmse = metrics["rmse"]
            patience_count = 0
            torch.save(model_ft.state_dict(), MODEL_DIR / "cnp_california_finetuned.pt")
        else:
            patience_count += 1
            if patience_count >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Final evaluation
    model_ft.load_state_dict(torch.load(MODEL_DIR / "cnp_california_finetuned.pt", weights_only=True))
    if len(cali_test_ds.groups) > 0:
        final = evaluate(model_ft, cali_test_ds, n_episodes=200)
        print(f"\n  Fine-tuned: RMSE={final['rmse']:.4f}, R²={final['r2']:.4f}, "
              f"σ={final['mean_sigma']:.4f}, coverage={final['coverage_95']:.3f}")

    # Per-metric evaluation
    print("\n  Per metric-type:")
    for mt_name in ["He", "pi"]:
        subset = cali_test[cali_test["metric_type"] == mt_name]
        if len(subset) < 4:
            continue
        sub_ds = WithinSpeciesCNPDataset(subset, processor, min_pops=2)
        if len(sub_ds.groups) > 0:
            m = evaluate(model_ft, sub_ds, n_episodes=50)
            print(f"    {mt_name}: n_species={subset['species'].nunique()}, "
                  f"RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}, σ={m['mean_sigma']:.4f}")

    # Save
    with open(MODEL_DIR / "feature_processor.pkl", "wb") as f:
        pickle.dump(processor, f)
    print(f"\nSaved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 6c: CNP-projected genetic diversity under end-of-century climate.

`data/processed/vulnerability_scores_future.csv` on disk predates the current
06_future_climate.py and carries only the exposure columns, so the projected
diversity columns are missing downstream. This script recomputes just that block
— the same code path as step 6 — and writes it to its own file rather than
overwriting any step 6 output.

Outputs:
  data/processed/future_diversity.csv
    species, Latitude, Longitude, pred_mu_future, pred_sigma_future,
    diversity_norm_future, sigma_norm_future
  (row-aligned with data/processed/vulnerability_scores.csv)
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT))
_f = __import__("06_future_climate")
_p = __import__("04_predict_california")
_m = __import__("03_train_model")

# models/feature_processor.pkl was pickled from 03_train_model.py running as
# __main__, so these names have to exist in this script's namespace to unpickle.
FeatureProcessor = _m.FeatureProcessor
ConditionalNeuralProcess = _m.ConditionalNeuralProcess


def main():
    current = pd.read_csv(PROC / "vulnerability_scores.csv")
    print(f"Records: {len(current)}, species: {current['species'].nunique()}")

    coast_lats, coast_lons = _p.make_coastal_grid()
    tree = cKDTree(np.column_stack([coast_lats, coast_lons]))

    print("\nFetching future marine grid (Bio-ORACLE SSP5-8.5, avg 2080+2090)...")
    marine_grid = pd.DataFrame({"Latitude": coast_lats, "Longitude": coast_lons})
    marine_grid = _f.extract_future_marine_avg(marine_grid)

    print("Building future environment ensemble (5 GCMs)...")
    future_grid = _f.build_future_env_ensemble(coast_lats, coast_lons, marine_grid)
    swap_cols = [c for c in future_grid.columns if c not in ["Latitude", "Longitude"]]

    model, processor = _p.load_model_and_processor()
    # Context must carry LULC columns so the LULC model sees them (not zero-filled)
    cali_df = _m.filter_ca_species(pd.read_csv(PROC / _p.CALI_FILE))

    global_df = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)
    HE_LO, HE_HI = global_df["gen_div"].quantile(0.02), global_df["gen_div"].quantile(0.98)
    pi_vals = cali_df[cali_df["metric_type"] == "pi"]["gen_div"]
    PI_LO, PI_HI = pi_vals.quantile(0.02), pi_vals.quantile(0.98)

    mu_out = np.full(len(current), np.nan)
    sigma_out = np.full(len(current), np.nan)

    print("\nPredicting under future climate:")
    for sp in current["species"].unique():
        mask = (current["species"] == sp).values
        rows = current[mask]
        lats, lons = rows["Latitude"].values, rows["Longitude"].values

        context = _p.prepare_context(cali_df[cali_df["species"] == sp], processor)
        target = _p.make_target_df(lats, lons, sp, rows["metric_type"].iloc[0], cali_df)
        _, gi = tree.query(np.column_stack([lats, lons]))
        for col in swap_cols:
            if col in target.columns:
                target[col] = future_grid[col].values[gi]

        mu, sigma = _p.predict_at_locations(model, context, target, processor)
        mu_out[mask], sigma_out[mask] = mu, sigma
        print(f"  {sp[:30]:32s} {rows['pred_mu'].mean():.4f} → {mu.mean():.4f} "
              f"({mu.mean() - rows['pred_mu'].mean():+.4f})")

    # Score scale: the SAME cross-species reference step 5 uses, so present and
    # future diversity are directly comparable in the vulnerability score.
    div_norm = np.full(len(current), np.nan)
    he = (current["metric_type"] == "He").values
    pi = (current["metric_type"] == "pi").values
    div_norm[he] = np.clip((mu_out[he] - HE_LO) / (HE_HI - HE_LO), 0, 1)
    div_norm[pi] = np.clip((mu_out[pi] - PI_LO) / (PI_HI - PI_LO), 0, 1)

    # Display scale: within-species, spanning both time slices, for plots only.
    # Present and future have to share one within-species scale, and it must cover
    # both — the spread across a species' range is far narrower than the change it
    # undergoes by 2100, so scaling on the present alone pushes every future value
    # off the ends (He clipping to 0, pi to 1).
    rows = []
    for sp in current["species"].unique():
        m = (current["species"] == sp).values
        both = np.concatenate([current.loc[m, "pred_mu"].values, mu_out[m]])
        rows.append({"species": sp,
                     "metric_type": current.loc[m, "metric_type"].iloc[0],
                     "div_lo": np.nanmin(both), "div_hi": np.nanmax(both)})
    scale = pd.DataFrame(rows).set_index("species")
    scale.to_csv(PROC / "diversity_scaling.csv")

    div_within = np.full(len(current), np.nan)
    for sp in current["species"].unique():
        m = (current["species"] == sp).values
        lo, hi = scale.loc[sp, "div_lo"], scale.loc[sp, "div_hi"]
        div_within[m] = (np.clip((mu_out[m] - lo) / (hi - lo), 0, 1)
                         if hi - lo > 1e-8 else 0.5)

    sigma_norm = np.zeros(len(current))
    for sp in current["species"].unique():
        mask = (current["species"] == sp).values
        v = sigma_out[mask]
        if v.max() - v.min() > 1e-8:
            sigma_norm[mask] = (v - v.min()) / (v.max() - v.min())

    out = current[["species", "metric_type", "Latitude", "Longitude"]].copy()
    out["pred_mu_future"] = mu_out
    out["pred_sigma_future"] = sigma_out
    out["diversity_norm_future"] = div_norm
    out["diversity_norm_within_future"] = div_within
    out["sigma_norm_future"] = sigma_norm
    out.to_csv(PROC / "future_diversity.csv", index=False)

    print(f"\nSaved: {PROC / 'future_diversity.csv'}")
    print(f"  diversity_norm (cross-species): current {current['diversity_norm'].mean():.4f} → "
          f"future {np.nanmean(div_norm):.4f} "
          f"({np.nanmean(div_norm) - current['diversity_norm'].mean():+.4f})")


if __name__ == "__main__":
    main()

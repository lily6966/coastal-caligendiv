#!/usr/bin/env python3
"""
Step 11: Per-GCM future genetic-diversity projections and historical-vs-future
comparison at species and community levels.

Unlike 06/06c (which average the five GCMs into one climate grid before a single
prediction), this predicts future diversity under EACH GCM separately — 1 km
terrestrial climate per GCM (climate_delta.sample_bio, matching the present-day
resolution), shared SSP5-8.5 future marine, present-day land use — then reports
the ensemble mean and the between-GCM standard deviation. Projected diversity
therefore carries genuine inter-model variation.

Outputs:
  data/processed/gcm_ensemble_diversity.csv
  figures/historical_vs_future_diversity.{pdf,png}
"""
import sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
_p = __import__("04_predict_california")
_f = __import__("06_future_climate")
import climate_delta as cd
FeatureProcessor = _m.FeatureProcessor
ConditionalNeuralProcess = _m.ConditionalNeuralProcess

BIOS = [1, 4, 5, 6, 7, 12, 13, 14, 15]
MARINE = ["sst_mean", "sst_max", "sst_range", "salinity_mean", "chl_mean", "o2_mean", "ph_mean"]
GCMS = cd.GCMS


def main():
    model, processor = _p.load_model_and_processor()
    cali = _m.filter_ca_species(pd.read_csv(PROC / _p.CALI_FILE))
    grid = pd.read_csv(PROC / "vulnerability_scores.csv").reset_index(drop=True)

    # cross-species normalization references (match step 5)
    g_he = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)["gen_div"]
    HE_LO, HE_HI = g_he.quantile(0.02), g_he.quantile(0.98)
    pi_v = cali[cali["metric_type"] == "pi"]["gen_div"]
    PI_LO, PI_HI = pi_v.quantile(0.02), pi_v.quantile(0.98)
    # cali reference stats for the climate_stress z-composite (as in env_present)
    ref = {c: (cali[c].mean(), cali[c].std()) for c in ["bio5", "bio14", "bio4"]}

    # shared SSP5-8.5 future marine at every grid point (one fetch)
    print("Fetching shared future marine (Bio-ORACLE SSP5-8.5, avg 2080+2090)...")
    fm = _f.extract_future_marine_avg(grid[["Latitude", "Longitude"]].copy())

    # per-GCM future terrestrial bios at every grid point (1 km, CA)
    print("Sampling per-GCM future terrestrial climate (1 km)...")
    lat = grid["Latitude"].values; lon = grid["Longitude"].values
    fut_bio = {g: {b: cd.sample_bio(b, lat, lon, gcm=g) for b in BIOS} for g in GCMS}

    # predict future diversity under each GCM, per species
    print("Predicting future diversity per GCM:")
    fut = {g: np.full(len(grid), np.nan) for g in GCMS}
    for sp in grid["species"].unique():
        idx = grid.index[grid["species"] == sp].values
        mt = grid.loc[idx[0], "metric_type"]
        obs = cali[cali["species"] == sp]
        context = _p.prepare_context(obs, processor)
        base = _p.make_target_df(lat[idx], lon[idx], sp, mt, cali)  # present env + LULC + categoricals
        for g in GCMS:
            tgt = base.copy()
            for b in BIOS:
                tgt[f"bio{b}"] = fut_bio[g][b][idx]
            for col in MARINE:
                if col in fm.columns:
                    tgt[col] = fm[col].values[idx]
            # recompute derived features from the future bios (cali-referenced z)
            tgt["temp_seasonality_norm"] = tgt["bio4"] / (tgt["bio7"] * 100 + 1)
            tgt["precip_extremity"] = tgt["bio13"] / (tgt["bio14"] + 1)
            z = {c: (tgt[c] - ref[c][0]) / ref[c][1] if ref[c][1] > 0 else 0.0 for c in ref}
            tgt["climate_stress"] = (z["bio5"] - z["bio14"] + z["bio4"]) / 3
            mu, _ = _p.predict_at_locations(model, context, tgt, processor)
            fut[g][idx] = mu
        print(f"  {sp[:30]:32s} present {grid.loc[idx,'pred_mu'].mean():.4f} -> "
              f"future {np.nanmean([fut[g][idx].mean() for g in GCMS]):.4f}")

    # assemble ensemble mean / SD
    out = grid[["species", "metric_type", "Latitude", "Longitude", "pred_mu"]].copy()
    out = out.rename(columns={"pred_mu": "present_mu"})
    fmat = np.column_stack([fut[g] for g in GCMS])
    out["future_mu_mean"] = fmat.mean(axis=1)
    out["future_mu_sd"] = fmat.std(axis=1)
    for g in GCMS:
        out[f"future_mu_{g}"] = fut[g]

    def dnorm(v, mt):
        lo, hi = (HE_LO, HE_HI) if mt == "He" else (PI_LO, PI_HI)
        return np.clip((v - lo) / (hi - lo), 0, 1)
    out["present_dnorm"] = [dnorm(v, m) for v, m in zip(out["present_mu"], out["metric_type"])]
    out["future_dnorm_mean"] = [dnorm(v, m) for v, m in zip(out["future_mu_mean"], out["metric_type"])]
    # per-GCM normalized, for community SD
    for g in GCMS:
        out[f"future_dnorm_{g}"] = [dnorm(v, m) for v, m in zip(out[f"future_mu_{g}"], out["metric_type"])]
    out.to_csv(PROC / "gcm_ensemble_diversity.csv", index=False)
    print(f"\nSaved: {PROC/'gcm_ensemble_diversity.csv'}")

    _make_figure(out)


def _make_figure(out):
    GCMS_ = GCMS
    # species-level means
    sp = out.groupby(["species", "metric_type"]).agg(
        present=("present_mu", "mean"),
        future=("future_mu_mean", "mean")).reset_index()
    # between-GCM SD of the species-mean future (species-level inter-model spread)
    permu = out.groupby("species")[[f"future_mu_{g}" for g in GCMS_]].mean()
    sp = sp.merge(permu.std(axis=1).rename("future_sd").reset_index(), on="species")
    sp["delta"] = sp["future"] - sp["present"]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.35, wspace=0.28)

    # (a) He species dumbbell   (b) pi species dumbbell
    for row, mt, lab in [(0, "He", "Expected heterozygosity (He)"),
                         (1, "pi", "Nucleotide diversity (π)")]:
        ax = fig.add_subplot(gs[row, 0])
        d = sp[sp["metric_type"] == mt].sort_values("present")
        y = np.arange(len(d))
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.plot([r["present"], r["future"]], [yi, yi], "-", color="#bbbbbb", lw=2, zorder=1)
        ax.scatter(d["present"], y, s=55, color="#1f77b4", zorder=3, label="Historical")
        ax.errorbar(d["future"], y, xerr=d["future_sd"], fmt="D", ms=6, color="#d62728",
                    ecolor="#d62728", elinewidth=1.2, capsize=3, zorder=3,
                    label="Future 2100 (5-GCM mean ± SD)")
        ax.set_yticks(y); ax.set_yticklabels([s[:24] for s in d["species"]], fontsize=8)
        ax.set_xlabel(lab); ax.grid(True, axis="x", alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"Species-level change — {mt}", fontsize=11)

    # (c) community latitudinal profile (normalized diversity, He+pi pooled)
    ax = fig.add_subplot(gs[:, 1])
    bins = np.linspace(out["Latitude"].min(), out["Latitude"].max(), 40)
    ctr = (bins[:-1] + bins[1:]) / 2
    pres, futm, futsd = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (out["Latitude"] >= lo) & (out["Latitude"] < hi)
        if m.sum() == 0:
            pres.append(np.nan); futm.append(np.nan); futsd.append(np.nan); continue
        pres.append(out.loc[m, "present_dnorm"].mean())
        per_g = [out.loc[m, f"future_dnorm_{g}"].mean() for g in GCMS_]
        futm.append(np.mean(per_g)); futsd.append(np.std(per_g))
    pres, futm, futsd = map(np.array, (pres, futm, futsd))
    ax.plot(ctr, pres, "-", color="#1f77b4", lw=2.5, label="Historical")
    ax.plot(ctr, futm, "-", color="#d62728", lw=2.5, label="Future 2100 (ensemble mean)")
    ax.fill_between(ctr, futm - futsd, futm + futsd, color="#d62728", alpha=0.2,
                    label="Between-GCM ±1 SD")
    ax.set_xlabel("Latitude (°N)"); ax.set_ylabel("Community mean normalized diversity (0–1)")
    ax.set_title("Community level — normalized diversity vs latitude", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # community summary
    cp = out["present_dnorm"].mean()
    cf = np.mean([out[f"future_dnorm_{g}"].mean() for g in GCMS_])
    csd = np.std([out[f"future_dnorm_{g}"].mean() for g in GCMS_])
    ax.text(0.03, 0.03, f"Community mean: {cp:.3f} → {cf:.3f} ± {csd:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))

    fig.suptitle("Historical vs projected 2100 genetic diversity (SSP5-8.5, per-GCM ensemble)",
                 fontsize=14, y=0.98)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
        fig.savefig(FIG / f"historical_vs_future_diversity.{ext}", bbox_inches="tight", **kw)
    print(f"Saved: {FIG/'historical_vs_future_diversity.pdf'}")
    print(f"  Community normalized diversity: {cp:.3f} -> {cf:.3f} ± {csd:.3f} (between-GCM SD)")


if __name__ == "__main__":
    main()

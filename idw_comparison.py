#!/usr/bin/env python3
"""
IDW baseline vs the CNP: within-species predictive skill and species-level
vulnerability-ranking agreement.

(1) Within-species leave-one-out skill: each observed population is predicted
    from the remaining populations of its species, by inverse-distance weighting
    (IDW, great-circle distance, power p) and by the CNP. Reported per metric as
    RMSE, R^2, and the within-species anomaly correlation.

(2) Vulnerability ranking: along the same coastal grid, diversity is predicted by
    IDW, normalized with the same references as the CNP, and combined with the
    CNP's exposure (total_stress) and uncertainty (sigma_norm) using the paper's
    index  V = (1 - D_norm) * S + 0.3 * U.  Species are ranked by mean V under each
    method and compared with Spearman rho and Kendall tau.

Outputs: data/processed/idw_vs_cnp_skill.csv, idw_vs_cnp_ranking.csv,
         figures/idw_vs_cnp_ranking.{pdf,png}
"""
import os
import sys, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).parent
# Processed-data directory. $CNP_PROC_DIR redirects it, so a verification run
# reads and writes inside a copy and cannot modify the real outputs.
PROC = Path(os.environ.get("CNP_PROC_DIR") or (ROOT / "data" / "processed"))
# Figure output directory. $CNP_FIG_DIR redirects it, so a verification run
# can regenerate every figure without overwriting the committed ones.
FIG = Path(os.environ.get("CNP_FIG_DIR") or (ROOT / "figures"))
sys.path.insert(0, str(ROOT))
_m = __import__("03_train_model")
_p = __import__("04_predict_california")
FeatureProcessor = _m.FeatureProcessor
IDW_POWER = 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def idw_predict(tlat, tlon, olat, olon, oval, p=IDW_POWER):
    """IDW estimate at each target point from observed (olat,olon,oval)."""
    out = np.empty(len(tlat))
    for i in range(len(tlat)):
        d = haversine_km(tlat[i], tlon[i], olat, olon)
        if np.any(d < 1e-6):
            out[i] = oval[d < 1e-6].mean()
        else:
            w = 1.0 / d ** p
            out[i] = np.sum(w * oval) / np.sum(w)
    return out


def within_species_r(df, pred_col):
    """Pooled Pearson r of within-species anomalies (deviation from species mean)."""
    a_obs, a_pred = [], []
    for _, g in df.groupby("species"):
        a_obs.extend(g["obs"] - g["obs"].mean())
        a_pred.extend(g[pred_col] - g[pred_col].mean())
    a_obs, a_pred = np.array(a_obs), np.array(a_pred)
    return float(np.corrcoef(a_obs, a_pred)[0, 1])


def main():
    obs = pd.read_csv(PROC / _p.CALI_FILE)
    obs = _m.filter_ca_species(obs, verbose=False)
    model, proc = _p.load_model_and_processor()

    # ── (1) within-species LOO: IDW vs CNP ──
    rows = []
    for sp, g in obs.groupby("species"):
        if len(g) < 3:
            continue
        mt = g["metric_type"].iloc[0]
        lat = g["Latitude"].values; lon = g["Longitude"].values; y = g["gen_div"].values
        # CNP leave-one-out (raw units)
        mu_cnp, _, y_cnp = _p.leave_one_out_eval(model, g, proc)
        # IDW leave-one-out
        for i in range(len(g)):
            keep = np.ones(len(g), bool); keep[i] = False
            pred = idw_predict(lat[i:i+1], lon[i:i+1], lat[keep], lon[keep], y[keep])[0]
            rows.append({"species": sp, "metric_type": mt, "obs": y[i],
                         "idw": pred, "cnp": mu_cnp[i] if len(mu_cnp) == len(g) else np.nan})
    loo = pd.DataFrame(rows).dropna(subset=["cnp"])
    loo.to_csv(PROC / "idw_vs_cnp_skill.csv", index=False)

    print("=" * 64)
    print("WITHIN-SPECIES LEAVE-ONE-OUT SKILL (IDW vs CNP)")
    print("=" * 64)
    def skill(d, col):
        y, mu = d["obs"].values, d[col].values
        rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
        r2 = float(1 - np.sum((y - mu) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))
        return rmse, r2
    for mt in ["He", "pi"]:
        d = loo[loo["metric_type"] == mt]
        for col in ["idw", "cnp"]:
            rmse, r2 = skill(d, col)
            wr = within_species_r(d, col)
            print(f"  {mt:3s} {col.upper():3s}: RMSE={rmse:.5f}  R2={r2:.3f}  within-species r={wr:.3f}")

    # ── (2) vulnerability ranking: IDW-diversity plugged into the paper's index ──
    v = pd.read_csv(PROC / "vulnerability_scores.csv")
    # reference ranges (match 05)
    gdiv = pd.read_csv(PROC / "global_train_env.csv", low_memory=False)["gen_div"]
    HE_LO, HE_HI = gdiv.quantile(0.02), gdiv.quantile(0.98)
    pi_obs = obs[obs["metric_type"] == "pi"]["gen_div"]
    PI_LO, PI_HI = pi_obs.quantile(0.02), pi_obs.quantile(0.98)

    v = v.copy()
    idw_div = np.full(len(v), np.nan)
    for sp, g in v.groupby("species"):
        o = obs[obs["species"] == sp]
        idx = g.index.values
        idw_div[np.searchsorted(v.index.values, idx)] = idw_predict(
            g["Latitude"].values, g["Longitude"].values,
            o["Latitude"].values, o["Longitude"].values, o["gen_div"].values)
    v["idw_pred"] = idw_div
    he = v["metric_type"] == "He"; pi = v["metric_type"] == "pi"
    v.loc[he, "idw_dnorm"] = ((v.loc[he, "idw_pred"] - HE_LO) / (HE_HI - HE_LO)).clip(0, 1)
    v.loc[pi, "idw_dnorm"] = ((v.loc[pi, "idw_pred"] - PI_LO) / (PI_HI - PI_LO)).clip(0, 1)
    # same exposure S and uncertainty U as the CNP, keep 0.3*U
    v["idw_vulnerability"] = (1 - v["idw_dnorm"]) * v["total_stress"] + 0.3 * v["sigma_norm"]

    rank = v.groupby("species").agg(
        cnp_vuln=("vulnerability", "mean"),
        idw_vuln=("idw_vulnerability", "mean"),
        metric=("metric_type", "first")).reset_index()
    rank["cnp_rank"] = rank["cnp_vuln"].rank(ascending=False).astype(int)
    rank["idw_rank"] = rank["idw_vuln"].rank(ascending=False).astype(int)
    rank["rank_shift"] = (rank["cnp_rank"] - rank["idw_rank"]).abs()
    rank = rank.sort_values("cnp_rank")
    rank.to_csv(PROC / "idw_vs_cnp_ranking.csv", index=False)

    rho, rp = spearmanr(rank["cnp_vuln"], rank["idw_vuln"])
    tau, tp = kendalltau(rank["cnp_vuln"], rank["idw_vuln"])
    print("\n" + "=" * 64)
    print("SPECIES VULNERABILITY RANKING: CNP vs IDW (V = (1-D)*S + 0.3*U)")
    print("=" * 64)
    print(f"  Spearman rho = {rho:.3f} (p={rp:.3g}) | Kendall tau = {tau:.3f} (p={tp:.3g})")
    print(f"  Max rank shift = {rank['rank_shift'].max()} | mean = {rank['rank_shift'].mean():.2f}")
    print(f"\n  {'species':26s} {'mt':3s} {'CNP#':>4s} {'IDW#':>4s} {'shift':>5s}")
    for _, r in rank.iterrows():
        print(f"  {r['species'][:26]:26s} {r['metric']:3s} {r['cnp_rank']:4d} {r['idw_rank']:4d} {int(r['rank_shift']):5d}")

    # ── figure: rank-rank ──
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for _, r in rank.iterrows():
        ax.plot([0, 1], [r["cnp_rank"], r["idw_rank"]], "-", color="grey", alpha=0.5, lw=1)
        ax.text(-0.02, r["cnp_rank"], r["species"][:22], ha="right", va="center", fontsize=7)
        ax.text(1.02, r["idw_rank"], r["species"][:22], ha="left", va="center", fontsize=7)
    ax.scatter([0]*len(rank), rank["cnp_rank"], s=30, color="steelblue", zorder=3)
    ax.scatter([1]*len(rank), rank["idw_rank"], s=30, color="coral", zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CNP", "IDW"]); ax.set_xlim(-0.6, 1.6)
    ax.invert_yaxis(); ax.set_ylabel("Vulnerability rank (1 = most vulnerable)")
    ax.set_title(f"Species vulnerability ranking: CNP vs IDW\nSpearman ρ={rho:.2f}, Kendall τ={tau:.2f}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
        plt.savefig(FIG / f"idw_vs_cnp_ranking.{ext}", bbox_inches="tight", **kw)
    print(f"\nSaved: figures/idw_vs_cnp_ranking.pdf, "
          f"{PROC/'idw_vs_cnp_skill.csv'}, {PROC/'idw_vs_cnp_ranking.csv'}")


if __name__ == "__main__":
    main()

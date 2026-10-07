"""Step 32 - tuned XGBoost as the thesis's main model: held-out metrics, the
environmental ablation, SHAP and the per-listing examples, all on XGBoost.

Decision (user, 2026-09-23): the thesis reports tuned XGBoost (highest
held-out R2 in the step31 comparison) as its main model. The web app keeps
serving the tuned Random Forest, and the thesis says so. The RF analyses
(step25, step26) stay on disk as the record for the served model.

Held fixed from step31: the 6,167-row frame, the random 80/20 split (seed 42),
the out-of-fold target encoding, 50 features (six environmental and spatial
variables + NDVI), and the notebook's tuned XGBoost hyperparameters.

  1. held-out metrics, seed 42                    -> the headline numbers
  2. ablation, random split, seeds 42/1/7/123/2024:
       env7 (all seven) vs no_env vs env6 (no NDVI)
     same decision rule as step25 (|mean| > 2 SD and all seeds same sign)
  3. SHAP (TreeExplainer, exact for trees) on the 1,234 held-out rows
  4. two per-listing examples, chosen by step26's rule (lowest- and
     highest-HAND test division, median-priced listing)

Outputs: results/xgb_main/  (metrics.json, ablation.csv, shap_*.csv, figures,
         shap_examples.md)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step16_tuned_ablation_repeated as s16          # noqa: E402
import step25_env9_ablation as s25                    # noqa: E402
import step26_env9_shap as s26                        # noqa: E402

OUT = os.path.join(ROOT, "results", "xgb_main")
SEEDS = [42, 1, 7, 123, 2024]
XGB = dict(subsample=1.0, reg_lambda=2.0, n_estimators=500, max_depth=8,
           learning_rate=0.03, colsample_bytree=0.6, n_jobs=-1, verbosity=0)
ENV7 = s16.HAZARD_COLS + ["ndvi"]


def split(X, y, groups, cols, seed):
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=seed)
    oof, enc = s16.target_encode_oof(groups.iloc[itr], y.iloc[itr], groups.iloc[ite], seed)
    Xtr, Xte = X.iloc[itr][cols].copy(), X.iloc[ite][cols].copy()
    Xtr["Address_target_enc"] = oof.values
    Xte["Address_target_enc"] = enc.values
    return Xtr, Xte, y.iloc[itr], y.iloc[ite], ite


def fit(Xtr, ytr, seed):
    return XGBRegressor(random_state=seed, **XGB).fit(Xtr, ytr)


def metrics(yt, yp):
    rt, rp = np.exp(yt), np.exp(yp)
    ape = np.abs(rp - rt) / rt
    return dict(r2_pct=r2_score(yt, yp) * 100, rmse_log=float(np.sqrt(mean_squared_error(yt, yp))),
                mape_pct=float(ape.mean() * 100), medape_pct=float(np.median(ape) * 100),
                mae_rs=float(mean_absolute_error(rt, rp)))


def main():
    s26.OUT = OUT                                     # step26's figure writer targets this
    os.makedirs(OUT, exist_ok=True)
    X, y, groups, meta, prov = s25.build_matrix()
    assert prov["rows_used"] == 6167
    allc = list(X.columns)
    keep = lambda env: [c for c in allc if c not in s25.EXISTING + s25.NEW or c in env]
    cfg = {"env7": keep(ENV7), "env6_no_ndvi": keep(s16.HAZARD_COLS), "no_env": keep([])}

    # ------------------------------------------------------------ 1. headline
    Xtr, Xte, ytr, yte, ite = split(X, y, groups, cfg["env7"], 42)
    m = fit(Xtr, ytr, 42)
    pred = m.predict(Xte)
    head = metrics(yte, pred)
    print("held-out, seed 42:", {k: round(v, 4) for k, v in head.items()}, flush=True)

    # ------------------------------------------------------------ 2. ablation
    rows = []
    for name, cols in cfg.items():
        for sd in SEEDS:
            a, b, c, d, _ = split(X, y, groups, cols, sd)
            r2 = r2_score(d, fit(a, c, sd).predict(b)) * 100
            rows.append(dict(config=name, seed=sd, r2_pct=r2))
            print(f"  {name:13s} seed {sd:>4}  R2 {r2:6.2f}", flush=True)
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(OUT, "ablation_runs.csv"), index=False, float_format="%.4f")
    piv = ab.pivot(index="seed", columns="config", values="r2_pct").loc[SEEDS]
    deltas = {}
    for a_, b_ in (("no_env", "env7"), ("env6_no_ndvi", "env7")):
        d = piv[a_] - piv[b_]
        sdv = float(d.std(ddof=1))
        same = int((np.sign(d) == np.sign(d.mean())).sum())
        deltas[f"{a_} minus {b_}"] = dict(mean_pp=round(float(d.mean()), 4), two_sd_pp=round(2 * sdv, 4),
                                          same_sign=f"{same} of 5",
                                          detectable=bool(abs(d.mean()) > 2 * sdv and same == 5))
    summ = {k: dict(mean=round(float(v.mean()), 4), sd=round(float(v.std(ddof=1)), 4))
            for k, v in piv.items()}
    print(summ, deltas, flush=True)

    # --------------------------------------------------------------- 3. SHAP
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(Xte)
    base = float(np.ravel(ex.expected_value)[0])
    add_err = float(np.abs(base + sv.sum(1) - pred).max())
    feats = list(Xte.columns)
    g = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)})
    g["block"] = [s26.block_of(c, ENV7) for c in feats]
    g["share_pct"] = 100 * g.mean_abs_shap / g.mean_abs_shap.sum()
    g["rank"] = g.mean_abs_shap.rank(ascending=False).astype(int)
    g = g.sort_values("mean_abs_shap", ascending=False)
    g.to_csv(os.path.join(OUT, "shap_global.csv"), index=False, float_format="%.6f")
    blocks = g.groupby("block").share_pct.sum().sort_values(ascending=False)
    blocks.to_csv(os.path.join(OUT, "shap_blocks.csv"), float_format="%.4f")
    direc = pd.DataFrame([dict(variable=c, rank=int(g.set_index("feature").loc[c, "rank"]),
                               spearman_value_vs_shap=float(spearmanr(Xte[c], sv[:, feats.index(c)])[0]))
                          for c in ENV7])
    direc.to_csv(os.path.join(OUT, "shap_env_direction.csv"), index=False, float_format="%.4f")
    print(blocks.round(2).to_string(), "\n", direc.round(3).to_string(index=False), flush=True)

    jj = [feats.index(c) for c in ENV7]
    shap.summary_plot(sv[:, jj], Xte[ENV7], feature_names=ENV7, show=False, plot_size=(8, 4.5))
    plt.title("SHAP values - environmental and spatial variables (tuned XGBoost)\n"
              "(log price per perch; held-out listings)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_xgb_shap_beeswarm.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------ 4. examples
    rows_t = meta.iloc[ite].reset_index(drop=True).assign(
        test_row=np.arange(len(ite)), log_pred=pred, pred_rs=np.exp(pred),
        env_contrib=sv[:, jj].sum(1))
    div_hand = Xte.assign(Address=rows_t.Address.values).groupby("Address")["hand_m"].first()
    pct = lambda v: float((np.abs(rows_t.env_contrib) <= abs(v)).mean() * 100)
    md = ["# Per-listing SHAP examples - tuned XGBoost (thesis main model)", "",
          f"Held-out R² {head['r2_pct']:.2f}%. Base value Rs {np.exp(base):,.0f} per perch. "
          "Examples chosen by the rule fixed in step26 (lowest / highest HAND test division, "
          "median-priced listing). Associations, not causal effects.", ""]
    ex_out = {}
    for lab, addr in (("A", div_hand.idxmin()), ("B", div_hand.idxmax())):
        sub = rows_t[rows_t.Address == addr].sort_values("log_pred")
        r = sub.iloc[(len(sub) - 1) // 2]
        i = int(r.test_row)
        phi = pd.Series(sv[i], index=feats)
        grp = phi.groupby([s26.block_of(c, ENV7) for c in feats]).sum()
        steps = [(k, grp[k]) for k in ("division location (target encoding)", "distance from Fort",
                                       "amenities and access", "plot size and land type")]
        steps += [(c, phi[c]) for c in ENV7]
        s26.waterfall(lab, addr, base, steps, float(r.log_pred), float(r["Price per Perch"]),
                      Xte.iloc[i], ENV7)
        e_tot = float(phi[ENV7].sum())
        ex_out[lab] = dict(division=addr, asking=float(r["Price per Perch"]), predicted=float(r.pred_rs),
                           env_total_log=e_tot, env_total_pct=float(np.expm1(e_tot) * 100),
                           percentile=pct(e_tot))
        md += [f"## Example {lab} - {addr.title()}", "",
               f"Asking Rs {r['Price per Perch']:,.0f}; predicted Rs {r.pred_rs:,.0f}.", "",
               "| component | value | SHAP (log) | effect |", "|---|---|---|---|"]
        for k, v in steps:
            val = f"{Xte.iloc[i][k]:.3f}" if k in ENV7 else ""
            md.append(f"| {k} | {val} | {v:+.4f} | {np.expm1(v) * 100:+.2f}% |")
        md += [f"| **environmental block** | | **{e_tot:+.4f}** | **{np.expm1(e_tot) * 100:+.2f}%** |", "",
               f"Larger than in {pct(e_tot):.0f}% of held-out listings.", ""]
    med = float(np.median(np.abs(rows_t.env_contrib)))
    md += [f"Typical listing: median |block| {np.expm1(med) * 100:.2f}% of price.", ""]
    open(os.path.join(OUT, "shap_examples.md"), "w", encoding="utf-8").write("\n".join(md))
    for lab in ("A", "B"):                             # step26 names them fig_env9_*
        src = os.path.join(OUT, f"fig_env9_example_{lab}.png")
        if os.path.exists(src):
            os.replace(src, os.path.join(OUT, f"fig_xgb_example_{lab}.png"))

    json.dump(dict(model="XGBRegressor", params=XGB, n_features=len(feats), heldout=head,
                   ablation=summ, deltas=deltas, shap_blocks=blocks.round(3).to_dict(),
                   shap_additivity_max_err=add_err, examples=ex_out,
                   typical_env_pct=float(np.expm1(med) * 100)),
              open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(ex_out, indent=1))


if __name__ == "__main__":
    main()

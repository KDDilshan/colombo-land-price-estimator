"""Step 11 - recompute every metric including MAE, and per-fold metrics for spatial CV.

WHAT THIS FIXES
  Chapter 4 records two metric gaps:
    4.3  MAE is not computed anywhere in the project, though objective 1.5.4
         and RQ4 both name it among the four evaluation metrics.
    4.4  No per-fold RMSE or MAPE exists for any spatial CV run; both
         8.7_spatial_cv rows of model_results_log.csv have empty rmse_log and
         mape_pct fields, and step7_hazard_ablation.py records R2 only.

WHY MAE CANNOT BE ADDED TO model_results_log.csv AS A COLUMN
  The brief asked for MAE to be recomputed "on the saved predictions" for every
  run. THERE ARE NO SAVED PREDICTIONS. The notebook
  `model test/random_forest_land_price_model.ipynb` computes metrics inline and
  writes only the aggregates to the log; no y_pred array is persisted anywhere
  in the repository, and no model object is pickled.

  MAE therefore cannot be attached to the existing logged rows. Doing so would
  imply the logged run produced it, which is false. Instead this script re-runs
  the pipeline and writes a SEPARATE file whose rows are labelled as a
  re-execution, so the two can be compared but not confused.

  For the same reason, the mae_log values here belong to THIS run. Section 4.4.1
  of Chapter 4 established that the random-split figures reproduce to within
  about 0.1 pp but the spatial figure does not (49.36 measured against 53.52
  logged), so random-split MAE can be treated as a close estimate of the
  logged run's MAE while spatial MAE cannot.

DELIBERATE DEVIATION - hyperparameters are not re-searched
  Tuned models are instantiated from the notebook's printed best_params_, as in
  src/step8_shap_and_figures.py, so the fitted models are the notebook's
  selected models rather than a fresh search under a different library version.

SCOPE
  Two datasets are processed: final_dataset.csv (31 cols, no hazard) and
  final_dataset_hazard.csv (37 cols). The third block of
  model_results_log.csv - final_dataset_hazard_with_ratnapura.csv, 48 of its 70
  rows - is EXCLUDED. That file is 7,034 real Colombo rows plus 61 real
  Ratnapura rows plus 1,000 SYNTHETIC rows, and is out of scope for a
  Colombo-only study.

RESUMABILITY
  One dataset per invocation. Progress is cached under
  results/metrics/.cache/. Re-run until it prints DONE.

Outputs
  results/metrics/recomputed_metrics.csv     one row per stage/model, incl. mae
  results/metrics/spatial_cv_per_fold.csv    per-fold R2, RMSE, MAE, MAPE
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBRegressor

import lightgbm as _lgb
import xgboost as _xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUTDIR = os.path.join(ROOT, "results", "metrics")
CACHE = os.path.join(OUTDIR, ".cache")
os.makedirs(CACHE, exist_ok=True)

RANDOM_STATE = 42
DATASETS = ["final_dataset.csv", "final_dataset_hazard.csv"]
HAZARD_COLS = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
               "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]

RF_ORIG_BEST = dict(n_estimators=200, min_samples_split=4, min_samples_leaf=1,
                    max_features=None, max_depth=20)
RF_ENR_BEST = dict(n_estimators=300, min_samples_split=4, min_samples_leaf=1,
                   max_features=None, max_depth=20)
XGB_BEST = dict(subsample=1.0, reg_lambda=2.0, n_estimators=500, max_depth=8,
                learning_rate=0.03, colsample_bytree=0.6)
LGB_BEST = dict(subsample=0.8, num_leaves=63, n_estimators=300, max_depth=-1,
                learning_rate=0.03, colsample_bytree=0.8)

STARTED = dt.datetime.now()
ENV = (f"python={platform.python_version()} sklearn={sklearn.__version__} "
       f"xgboost={_xgb.__version__} lightgbm={_lgb.__version__} "
       f"numpy={np.__version__} pandas={pd.__version__}")


def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, group="Address", k=2):
    g = d.groupby(group)[col]
    mean, sd = g.transform("mean"), g.transform("std")
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], int((~keep).sum())


def target_encode_oof(tg, tt, vg, n_splits=5, smoothing=10,
                      random_state=RANDOM_STATE):
    gm = tt.mean()
    oof = pd.Series(index=tg.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for a, b in kf.split(tg):
        st = tt.iloc[a].groupby(tg.iloc[a]).agg(["mean", "count"])
        sm = (st["mean"] * st["count"] + gm * smoothing) / (st["count"] + smoothing)
        oof.iloc[b] = tg.iloc[b].map(sm).fillna(gm).values
    sf = tt.groupby(tg).agg(["mean", "count"])
    smf = (sf["mean"] * sf["count"] + gm * smoothing) / (sf["count"] + smoothing)
    return oof, vg.map(smf).fillna(gm)


def metrics(y_true, pred):
    """All four metrics. MAE is reported in BOTH log units and rupees."""
    yt = np.asarray(y_true, dtype=float)
    pr = np.asarray(pred, dtype=float)
    return {
        "r2_pct": r2_score(yt, pr) * 100,
        "rmse_log": float(np.sqrt(mean_squared_error(yt, pr))),
        "mae_log": float(mean_absolute_error(yt, pr)),
        "mae_rs_per_perch": float(mean_absolute_error(np.exp(yt), np.exp(pr))),
        "mape_pct": mean_absolute_percentage_error(np.exp(yt), np.exp(pr)) * 100,
    }


def run_dataset(name):
    path = os.path.join(PROC, name)
    df_full = pd.read_csv(path, low_memory=False)
    rows, folds = [], []
    tag = f"dataset={name} shape={df_full.shape[0]}x{df_full.shape[1]}"
    print(f"\n=== {name}  {df_full.shape}")

    def rec(stage, model, y, pred, note=""):
        m = metrics(y, pred)
        rows.append({"dataset": name, "stage": stage, "model": model,
                     **{k: round(v, 6) for k, v in m.items()}, "notes": note})
        print(f"  {stage:22s} {model:22s} R2={m['r2_pct']:6.2f}  "
              f"RMSE={m['rmse_log']:.4f}  MAE(log)={m['mae_log']:.4f}  "
              f"MAPE={m['mape_pct']:6.2f}")

    # ---- stage A: original features, k=2 trim
    df, n1 = drop_per_group(df_full.copy(), "Price per Perch")
    df, n2 = drop_per_group(df, "Land_size(Perches)")
    y = np.log(df["Price per Perch"])
    dm = pd.get_dummies(df["Land_type"], prefix="Land_type").astype(int)
    am = [c for c in df.columns if c.startswith(("min_dist", "count_"))]
    hz = [c for c in HAZARD_COLS if c in df.columns]
    X = pd.concat([df[["Land_size(Perches)", "Distance from fort"] + am + hz], dm], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20,
                                          random_state=RANDOM_STATE)
    rec("5_baseline", "RandomForest", yte,
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                              n_jobs=-1).fit(Xtr, ytr).predict(Xte),
        f"{len(X.columns)} features; k=2 trim -> {len(df)} rows")
    rec("6_tuned", "RandomForest", yte,
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                              **RF_ORIG_BEST).fit(Xtr, ytr).predict(Xte),
        "notebook best_params_, not re-searched")

    # ---- stage B: enriched, k=1.5 trim
    df2, m1 = drop_per_group(df_full.copy(), "Price per Perch", k=1.5)
    df2, m2 = drop_per_group(df2, "Land_size(Perches)", k=1.5)
    y2 = np.log(df2["Price per Perch"])
    cc = [c for c in df2.columns if c.startswith("count_")]
    mc = [c for c in df2.columns if c.startswith("min_dist")]
    df2["total_amenity_count"] = df2[cc].sum(axis=1)
    df2["min_dist_any_amenity"] = df2[mc].min(axis=1)
    df2["school_count_total"] = (df2["count_govtschools_A"]
                                 + df2["count_semigovtschools"]
                                 + df2["count_intlschools"])
    df2["health_count_total"] = (df2["count_Govt_Hospitals"]
                                 + df2["count_Pvt_Hospital"]
                                 + df2["count_Pvt_Med_Centers"])
    df2["finance_count_total"] = (df2["count_banks_within_2km"]
                                  + df2["count_FinanceCompanies_within_2km"])
    eng = ["total_amenity_count", "min_dist_any_amenity", "school_count_total",
           "health_count_total", "finance_count_total"]
    dm2 = pd.get_dummies(df2["Land_type"], prefix="Land_type").astype(int)
    am2 = [c for c in df2.columns
           if c.startswith(("min_dist", "count_")) and c not in eng]
    hz2 = [c for c in HAZARD_COLS if c in df2.columns]
    X2 = pd.concat([df2[["Land_size(Perches)", "Distance from fort"]
                        + am2 + hz2 + eng], dm2], axis=1)
    X2["_Address"] = df2["Address"].values
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=0.20,
                                              random_state=RANDOM_STATE)
    atr, ate = X2tr["_Address"], X2te["_Address"]
    oof, tenc = target_encode_oof(atr, y2tr, ate)
    X2tr = X2tr.drop(columns=["_Address"]).copy()
    X2te = X2te.drop(columns=["_Address"]).copy()
    X2tr["Address_target_enc"] = oof.values
    X2te["Address_target_enc"] = tenc.values
    X2tr.columns = [clean_name(c) for c in X2tr.columns]
    X2te.columns = [clean_name(c) for c in X2te.columns]
    nfeat = len(X2tr.columns)

    cands = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    }
    for nm, mdl in cands.items():
        rec("8.4_enriched_untuned", nm, y2te, mdl.fit(X2tr, y2tr).predict(X2te),
            f"{nfeat} features; k=1.5 trim -> {len(df2)} rows")

    rf_b = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **RF_ENR_BEST).fit(X2tr, y2tr)
    p_rf = rf_b.predict(X2te); rec("8.5_enriched_tuned", "RandomForest", y2te, p_rf)
    p_xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                         **XGB_BEST).fit(X2tr, y2tr).predict(X2te)
    rec("8.5_enriched_tuned", "XGBoost", y2te, p_xgb)
    p_lgb = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                          **LGB_BEST).fit(X2tr, y2tr).predict(X2te)
    rec("8.5_enriched_tuned", "LightGBM", y2te, p_lgb)
    ens = (p_rf + p_xgb + p_lgb) / 3
    rec("8.6_ensemble", "RF+XGB+LGB avg", y2te, ens, "best random-split result")

    # ---- stage C: spatial CV, PER-FOLD metrics
    Xfull = pd.concat([X2tr.drop(columns=["Address_target_enc"]),
                       X2te.drop(columns=["Address_target_enc"])], axis=0)
    yfull = pd.concat([y2tr, y2te], axis=0)
    afull = pd.concat([atr, ate], axis=0)
    Xfull = Xfull.loc[yfull.index]
    gkf = GroupKFold(n_splits=5)
    fold_m = []
    for i, (tr, va) in enumerate(gkf.split(Xfull, yfull, groups=afull), 1):
        Xa, Xb = Xfull.iloc[tr].copy(), Xfull.iloc[va].copy()
        ya, yb = yfull.iloc[tr], yfull.iloc[va]
        ga, gb = afull.iloc[tr], afull.iloc[va]
        o, e = target_encode_oof(ga, ya, gb)
        Xa["Address_target_enc"] = o.values
        Xb["Address_target_enc"] = e.values
        mdl = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                    n_jobs=-1).fit(Xa, ya)
        m = metrics(yb, mdl.predict(Xb))
        fold_m.append(m)
        folds.append({"dataset": name, "spec": "notebook_cell17_RF300_untuned",
                      "fold": i, "n_train_rows": len(tr), "n_val_rows": len(va),
                      "n_val_divisions": int(gb.nunique()),
                      **{k: round(v, 6) for k, v in m.items()}})
        print(f"  spatial fold {i}: R2={m['r2_pct']:6.2f}  RMSE={m['rmse_log']:.4f}  "
              f"MAE(log)={m['mae_log']:.4f}  MAPE={m['mape_pct']:6.2f}")
    agg = {k: float(np.mean([f[k] for f in fold_m])) for k in fold_m[0]}
    folds.append({"dataset": name, "spec": "notebook_cell17_RF300_untuned",
                  "fold": "mean", "n_train_rows": "", "n_val_rows": len(yfull),
                  "n_val_divisions": int(afull.nunique()),
                  **{k: round(v, 6) for k, v in agg.items()}})
    rows.append({"dataset": name, "stage": "8.7_spatial_cv",
                 "model": "RandomForest (NOT an ensemble - see note)",
                 **{k: round(v, 6) for k, v in agg.items()},
                 "notes": "GroupKFold(5) by Address, mean of per-fold metrics; "
                          "model_results_log.csv labels this stage RF+XGB+LGB avg "
                          "but notebook cell 17 fits a RandomForest only"})
    print(f"  spatial MEAN : R2={agg['r2_pct']:6.2f}  RMSE={agg['rmse_log']:.4f}  "
          f"MAE(log)={agg['mae_log']:.4f}  MAPE={agg['mape_pct']:6.2f}")
    return rows, folds, tag


def header(extra):
    lines = [f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step11_full_metrics.py",
             f"# env {ENV}",
             "# RE-EXECUTION, not the logged run. No predictions were ever saved by",
             "#   the notebook, so MAE could not be computed from stored output and",
             "#   the pipeline was re-run. Random-split figures reproduce the log to",
             "#   ~0.1 pp; the spatial figure does NOT (see Chapter 4 Section 4.4.1).",
             "# hyperparameters=notebook best_params_ instantiated directly",
             "# mae_log is MAE in log units; mae_rs_per_perch is MAE after",
             "#   back-transformation to rupees per perch",
             "# EXCLUDED: final_dataset_hazard_with_ratnapura.csv (48 of the 70 rows",
             "#   in model_results_log.csv) - contains 1,000 synthetic rows, out of scope"]
    return "\n".join(lines + [f"# {e}" for e in extra]) + "\n"


def write_csv(path, df, extra):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header(extra))
        df.to_csv(fh, index=False)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def main():
    state_p = os.path.join(CACHE, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"rows": [], "folds": [], "done": []}
    todo = [d for d in DATASETS if d not in state["done"]]
    if not todo:
        print("all datasets already processed; rewriting outputs from cache")
    for name in todo:
        r, f, _ = run_dataset(name)
        state["rows"] += r
        state["folds"] += f
        state["done"].append(name)
        json.dump(state, open(state_p, "w"), indent=1)
        break   # one dataset per invocation

    write_csv(os.path.join(OUTDIR, "recomputed_metrics.csv"),
              pd.DataFrame(state["rows"]),
              ["all four metrics for every stage of both Colombo datasets"])
    write_csv(os.path.join(OUTDIR, "spatial_cv_per_fold.csv"),
              pd.DataFrame(state["folds"]),
              ["per-fold R2, RMSE, MAE and MAPE for the spatial CV runs -",
               "  these did not previously exist for any fold"])
    remaining = [d for d in DATASETS if d not in state["done"]]
    if remaining:
        print(f"\nREMAINING: {remaining} - RE-RUN THIS SCRIPT")
        sys.exit(3)
    print("\n[DONE] both datasets processed")


if __name__ == "__main__":
    main()

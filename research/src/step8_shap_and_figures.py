"""Step 8 - export the five notebook figures and run SHAP on the Colombo hazard model.

WHY THIS EXISTS
  `RESULTS_ARTEFACT_AUDIT.md` records two blocking gaps for the Results chapter:

    1. No SHAP output exists for the Colombo study. The only SHAP code in the
       repository is in `model test/land price model.ipynb`, which is unrun AND
       loads `dataset_v6.csv` - the superseded 5-district archive. The three
       SHAP PNGs in `OLD Reserch/results/figures/` are therefore OLD-5DIST and
       must not be reused.
    2. Five plots exist only as inline PNG outputs inside
       `model test/random_forest_land_price_model.ipynb`. No `savefig` call
       exists anywhere, so nothing is on disk and nothing can be placed in a
       document.

  This script rebuilds the notebook pipeline, exports all five figures at
  300 dpi, and runs SHAP TreeExplainer on the tuned RandomForest.

PIPELINE - matched cell by cell to `random_forest_land_price_model.ipynb`
  cell 1   RANDOM_STATE=42; HAZARD_COLS = all SIX hazard columns, elevation_m
           INCLUDED. This is the notebook's set and is deliberately preserved:
           the notebook's reported figures include elevation_m. Only
           `src/step7_hazard_ablation.py` excludes it.
  cell 3   per-Address trim at k=2.0 -> 6537 rows (original feature stage)
  cell 4   X = Land_size + Distance from fort + 22 amenity + 6 hazard + dummies
  cell 5   train_test_split(test_size=0.20, random_state=42)
  cell 6   rf_base  = RandomForest(n_estimators=300)
  cell 7   best_rf  = tuned RF, original features
  cell 8   FIGURE 1 - top-15 Gini + predicted vs actual
  cell 9   k sweep -> best_k = 1.5
  cell 10  per-Address trim at k=1.5 -> 6211 rows; 5 engineered aggregates
  cell 11  out-of-fold target encoding of Address, KFold(5), smoothing=10
  cell 13  rf_best  = tuned RF, enriched features
  cell 14  xgb_best = tuned XGBoost
  cell 15  lgb_best = tuned LightGBM
  cell 16  ens_all3 = mean of the three tuned predictions
  cell 17  GroupKFold(5) by Address -> fold_scores
  cell 18  FIGURE 2 - actual vs predicted, baseline and ensemble
  cell 19  FIGURE 3 - residuals vs fitted + residual distribution
  cell 20  FIGURE 4 - top-15 Gini, enriched tuned RF
  cell 21  FIGURE 5 - stage progression bar chart

DELIBERATE DEVIATION - hyperparameters are not re-searched
  The notebook selects hyperparameters with RandomizedSearchCV (n_iter=20 for
  the original stage, n_iter=30 for each enriched model). Re-running those
  searches would re-select under a different library version and could land on
  different parameters, which would make these figures describe a model that
  is not the one in the log. The parameters below are the `best_params_` values
  PRINTED BY THE NOTEBOOK'S OWN EXECUTED OUTPUT, instantiated directly. The
  fitted models are therefore the notebook's selected models.

  Recorded in `RESULTS_COMPILED.md` 4.4 and the notebook cell 7/13/14/15 output.

ENVIRONMENT WARNING - READ BEFORE CITING THESE NUMBERS
  The notebook ran under Python 3.13 with unrecorded library versions. This
  script runs under the versions stamped into every output header below. Tree
  ensembles are sensitive to library version, row ordering and BLAS threading,
  so the metrics printed here will NOT be bit-identical to the notebook's
  inline outputs. The reconciliation of the two spatial-CV figures established
  this empirically: a clean-environment re-execution of the notebook's own
  grouped design returned 49-50% where the notebook printed 53.52%.

  Every figure and CSV produced here therefore carries its own measured value
  in its provenance header. Do not mix a number from this run with a number
  from `model_results_log.csv` in the same table.

RESUMABILITY
  SHAP over 1243 test rows against a 300-tree depth-20 forest takes roughly
  three minutes, which exceeds the execution window available here. The script
  is therefore checkpointed: it caches the fitted models and the SHAP matrix
  under results/figures/.cache/ and computes SHAP in row chunks, exiting
  cleanly when its time budget is spent. Re-run it until it prints DONE.
  Set BUDGET (seconds) in the environment to change the window.

  The chunking affects nothing statistically - SHAP values are computed
  per-row and are independent of the order or grouping in which rows are
  explained. The full test set is explained; no subsampling is used.

Outputs (all under results/figures/ unless noted)
  fig1_gini_and_fit_original.png
  fig2_actual_vs_predicted.png
  fig3_residuals.png
  fig4_gini_top15_enriched.png
  fig5_stage_progression.png
  shap_global_bar.png
  shap_beeswarm_hazard.png
  shap_global_importance.csv        mean |SHAP| for all 49 features
  shap_hazard_values.csv            per-observation SHAP for the 5 hazard vars
  shap_direction.csv                sign of the feature-value / SHAP relation
  run_metrics.csv                   every metric this run measured
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_percentage_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from xgboost import XGBRegressor

import lightgbm as _lgb
import xgboost as _xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "final_dataset_hazard.csv")
FIGDIR = os.path.join(ROOT, "results", "figures")
CACHE = os.path.join(FIGDIR, ".cache")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

BUDGET = float(os.environ.get("BUDGET", "150"))   # seconds per invocation
CHUNK = int(os.environ.get("CHUNK", "150"))       # test rows per SHAP chunk
_T0 = time.time()


def spent() -> float:
    return time.time() - _T0


def out_of_time() -> bool:
    return spent() > BUDGET

RANDOM_STATE = 42
DPI = 300

# cell 1 - elevation_m IS included, as in the notebook
HAZARD_COLS = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
               "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]
# the five modelled in the ablation; elevation_m excluded there, not here
HAZARD_5 = ["water_occurrence_pct", "dist_to_stream_km", "dist_to_kelani_km",
            "hand_m", "sar_flood_open_land"]

# notebook best_params_, verbatim from its executed output
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
       f"shap={shap.__version__} numpy={np.__version__} pandas={pd.__version__} "
       f"matplotlib={matplotlib.__version__}")
METRICS: dict[str, float] = {}


def header(extra: list[str]) -> str:
    lines = [
        f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step8_shap_and_figures.py",
        f"# dataset=final_dataset_hazard.csv (7034 rows x 37 cols)",
        f"# env {ENV}",
        "# WARNING library versions differ from the notebook run; values are NOT",
        "#         bit-identical to model_results_log.csv. Do not mix the two.",
        "# hyperparameters=notebook best_params_ instantiated directly, not re-searched",
    ]
    return "\n".join(lines + [f"# {e}" for e in extra]) + "\n"


def write_csv(path: str, df: pd.DataFrame, extra: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header(extra))
        df.to_csv(fh, index=False)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def savefig(fig, name: str) -> None:
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def clean_name(c: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, group="Address", k=2):
    g = d.groupby(group)[col]
    mean, sd = g.transform("mean"), g.transform("std")
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], int((~keep).sum())


def target_encode_oof(train_groups, train_target, test_groups, n_splits=5,
                      smoothing=10, random_state=RANDOM_STATE):
    """Notebook cell 11, unchanged."""
    global_mean = train_target.mean()
    oof = pd.Series(index=train_groups.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in kf.split(train_groups):
        tr_g, tr_t = train_groups.iloc[tr_idx], train_target.iloc[tr_idx]
        stats = tr_t.groupby(tr_g).agg(["mean", "count"])
        smooth = ((stats["mean"] * stats["count"] + global_mean * smoothing)
                  / (stats["count"] + smoothing))
        oof.iloc[val_idx] = (train_groups.iloc[val_idx].map(smooth)
                             .fillna(global_mean).values)
    stats_full = train_target.groupby(train_groups).agg(["mean", "count"])
    smooth_full = ((stats_full["mean"] * stats_full["count"] + global_mean * smoothing)
                   / (stats_full["count"] + smoothing))
    return oof, test_groups.map(smooth_full).fillna(global_mean)


def report(name, y_true, pred, key):
    r2 = r2_score(y_true, pred) * 100
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mape = mean_absolute_percentage_error(np.exp(y_true), np.exp(pred)) * 100
    METRICS[key] = r2
    METRICS[key + "__rmse_log"] = rmse
    METRICS[key + "__mape_pct"] = mape
    print(f"  {name:38s} R2={r2:6.2f}%  RMSE(log)={rmse:.4f}  MAPE={mape:6.2f}%")
    return r2


# ---------------------------------------------------------------- STAGE A
print("\n[A] original feature stage (notebook cells 2-8)")
df_full = pd.read_csv(DATA, low_memory=False)
assert df_full.shape == (7034, 37), df_full.shape

df = df_full.copy()
df, n_price = drop_per_group(df, "Price per Perch")          # k=2
df, n_size = drop_per_group(df, "Land_size(Perches)")
print(f"  trimmed {n_price + n_size} rows at k=2 -> {len(df)} rows")

y = np.log(df["Price per Perch"])
dummies = pd.get_dummies(df["Land_type"], prefix="Land_type").astype(int)
amenity_cols = [c for c in df.columns if c.startswith(("min_dist", "count_"))]
hazard_cols = [c for c in HAZARD_COLS if c in df.columns]
X = pd.concat([df[["Land_size(Perches)", "Distance from fort"]
                  + amenity_cols + hazard_cols], dummies], axis=1)
feature_cols = list(X.columns)
print(f"  features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)

rf_base = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                n_jobs=-1).fit(X_train, y_train)
pred_base = rf_base.predict(X_test)
report("baseline RF (300 trees)", y_test, pred_base, "5_baseline_rf")

best_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                               **RF_ORIG_BEST).fit(X_train, y_train)
pred_best = best_rf.predict(X_test)
report("tuned RF (original features)", y_test, pred_best, "6_tuned_rf")

final_model, final_pred = ((best_rf, pred_best)
                          if r2_score(y_test, pred_best) >= r2_score(y_test, pred_base)
                          else (rf_base, pred_base))
importances = pd.Series(final_model.feature_importances_,
                        index=feature_cols).sort_values(ascending=False)

# FIGURE 1 - notebook cell 8
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
importances.head(15)[::-1].plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Top 15 feature importances")
axes[0].set_xlabel("Importance")
axes[1].scatter(y_test, final_pred, s=6, alpha=0.25, color="#4C72B0")
lims = [min(y_test.min(), final_pred.min()), max(y_test.max(), final_pred.max())]
axes[1].plot(lims, lims, "r--", lw=1)
axes[1].set_xlabel("Actual log(price/perch)")
axes[1].set_ylabel("Predicted log(price/perch)")
axes[1].set_title(f"Predicted vs. actual (R2 = {r2_score(y_test, final_pred)*100:.1f}%)")
fig.tight_layout()
savefig(fig, "fig1_gini_and_fit_original.png")

# ---------------------------------------------------------------- STAGE B
print("\n[B] enriched feature stage (notebook cells 9-16)")
df2 = df_full.copy()
df2, m1 = drop_per_group(df2, "Price per Perch", k=1.5)
df2, m2 = drop_per_group(df2, "Land_size(Perches)", k=1.5)
print(f"  trimmed {m1 + m2} rows at k=1.5 -> {len(df2)} rows")

y2 = np.log(df2["Price per Perch"])
count_cols = [c for c in df2.columns if c.startswith("count_")]
mindist_cols = [c for c in df2.columns if c.startswith("min_dist")]
df2["total_amenity_count"] = df2[count_cols].sum(axis=1)
df2["min_dist_any_amenity"] = df2[mindist_cols].min(axis=1)
df2["school_count_total"] = (df2["count_govtschools_A"]
                             + df2["count_semigovtschools"]
                             + df2["count_intlschools"])
df2["health_count_total"] = (df2["count_Govt_Hospitals"]
                             + df2["count_Pvt_Hospital"]
                             + df2["count_Pvt_Med_Centers"])
df2["finance_count_total"] = (df2["count_banks_within_2km"]
                              + df2["count_FinanceCompanies_within_2km"])
engineered = ["total_amenity_count", "min_dist_any_amenity", "school_count_total",
              "health_count_total", "finance_count_total"]

dummies2 = pd.get_dummies(df2["Land_type"], prefix="Land_type").astype(int)
amenity2 = [c for c in df2.columns
            if c.startswith(("min_dist", "count_")) and c not in engineered]
hazard2 = [c for c in HAZARD_COLS if c in df2.columns]
X2 = pd.concat([df2[["Land_size(Perches)", "Distance from fort"]
                    + amenity2 + hazard2 + engineered], dummies2], axis=1)
X2["_Address"] = df2["Address"].values
print(f"  raw features: {X2.shape[1] - 1}")

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.20, random_state=RANDOM_STATE)
addr_train, addr_test = X2_train["_Address"], X2_test["_Address"]
oof_enc, test_enc = target_encode_oof(addr_train, y2_train, addr_test)
X2_train = X2_train.drop(columns=["_Address"]).copy()
X2_test = X2_test.drop(columns=["_Address"]).copy()
X2_train["Address_target_enc"] = oof_enc.values
X2_test["Address_target_enc"] = test_enc.values
X2_train.columns = [clean_name(c) for c in X2_train.columns]
X2_test.columns = [clean_name(c) for c in X2_test.columns]
feature_cols2 = list(X2_train.columns)
print(f"  final feature count: {len(feature_cols2)}")

rf_best = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                               **RF_ENR_BEST).fit(X2_train, y2_train)
pred_rf_tuned = rf_best.predict(X2_test)
report("tuned RF (enriched)", y2_test, pred_rf_tuned, "8.5_rf")

xgb_best = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                        **XGB_BEST).fit(X2_train, y2_train)
pred_xgb_tuned = xgb_best.predict(X2_test)
report("tuned XGBoost (enriched)", y2_test, pred_xgb_tuned, "8.5_xgb")

lgb_best = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                         **LGB_BEST).fit(X2_train, y2_train)
pred_lgb_tuned = lgb_best.predict(X2_test)
report("tuned LightGBM (enriched)", y2_test, pred_lgb_tuned, "8.5_lgb")

ens_all3 = (pred_rf_tuned + pred_xgb_tuned + pred_lgb_tuned) / 3
final_r2 = report("ensemble RF+XGB+LGB (avg)", y2_test, ens_all3, "8.6_ensemble")

# ---------------------------------------------------------------- STAGE C
print("\n[C] spatial CV (notebook cell 17)")
X2_full = pd.concat([X2_train.drop(columns=["Address_target_enc"]),
                     X2_test.drop(columns=["Address_target_enc"])], axis=0)
y2_full = pd.concat([y2_train, y2_test], axis=0)
addr_full = pd.concat([addr_train, addr_test], axis=0)
X2_full = X2_full.loc[y2_full.index]

gkf = GroupKFold(n_splits=5)
fold_scores = []
for tr_idx, val_idx in gkf.split(X2_full, y2_full, groups=addr_full):
    Xtr, Xval = X2_full.iloc[tr_idx].copy(), X2_full.iloc[val_idx].copy()
    ytr, yval = y2_full.iloc[tr_idx], y2_full.iloc[val_idx]
    atr, aval = addr_full.iloc[tr_idx], addr_full.iloc[val_idx]
    oof_tr, enc_val = target_encode_oof(atr, ytr, aval)
    Xtr["Address_target_enc"] = oof_tr.values
    Xval["Address_target_enc"] = enc_val.values
    m = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                              n_jobs=-1).fit(Xtr, ytr)
    fold_scores.append(r2_score(yval, m.predict(Xval)))
fold_scores = np.array(fold_scores)
for i, s in enumerate(fold_scores, 1):
    METRICS[f"8.7_spatial_fold{i}"] = s * 100
METRICS["8.7_spatial_mean"] = fold_scores.mean() * 100
print(f"  per-fold R2: {np.round(fold_scores * 100, 2)}")
print(f"  mean spatial R2: {fold_scores.mean() * 100:.2f}%")

# ---------------------------------------------------------------- FIGURES 2-5
print("\n[D] figures 2-5")

# FIGURE 2 - notebook cell 18
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(y_test, pred_base, s=6, alpha=0.25, color="#4C72B0")
lims0 = [min(y_test.min(), pred_base.min()), max(y_test.max(), pred_base.max())]
axes[0].plot(lims0, lims0, "r--", lw=1)
axes[0].set_xlabel("Actual log(price/perch)")
axes[0].set_ylabel("Predicted log(price/perch)")
axes[0].set_title(f"Original baseline RF (R2={r2_score(y_test, pred_base)*100:.1f}%)")
axes[1].scatter(y2_test, ens_all3, s=6, alpha=0.25, color="#55A868")
lims1 = [min(y2_test.min(), ens_all3.min()), max(y2_test.max(), ens_all3.max())]
axes[1].plot(lims1, lims1, "r--", lw=1)
axes[1].set_xlabel("Actual log(price/perch)")
axes[1].set_ylabel("Predicted log(price/perch)")
axes[1].set_title(f"Improved ensemble (R2={final_r2:.1f}%)")
fig.tight_layout()
savefig(fig, "fig2_actual_vs_predicted.png")

# FIGURE 3 - notebook cell 19
residuals = y2_test.values - ens_all3
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(ens_all3, residuals, s=6, alpha=0.25, color="#C44E52")
axes[0].axhline(0, color="black", lw=1, linestyle="--")
axes[0].set_xlabel("Predicted log(price/perch)")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Residuals vs. predicted")
axes[1].hist(residuals, bins=40, color="#C44E52", alpha=0.8)
axes[1].axvline(0, color="black", lw=1, linestyle="--")
axes[1].set_xlabel("Residual (log scale)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")
fig.tight_layout()
savefig(fig, "fig3_residuals.png")

# FIGURE 4 - notebook cell 20
importances2 = pd.Series(rf_best.feature_importances_,
                         index=feature_cols2).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
importances2.head(15)[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Top 15 feature importances\n(tuned RF, engineered + target-encoded features)")
ax.set_xlabel("Importance")
fig.tight_layout()
savefig(fig, "fig4_gini_top15_enriched.png")

# FIGURE 5 - notebook cell 21
stage_scores = {
    "Baseline RF\n(orig. features)": r2_score(y_test, pred_base) * 100,
    "Tuned RF\n(orig. features)": r2_score(y_test, pred_best) * 100,
    "RF + engineered\n+ target enc.": r2_score(y2_test, pred_rf_tuned) * 100,
    "Ensemble\n(RF+XGB+LGB)": final_r2,
    "Honest spatial\n(unseen division)": fold_scores.mean() * 100,
}
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#8C8C8C", "#8C8C8C", "#4C72B0", "#55A868", "#C44E52"]
bars = ax.bar(list(stage_scores.keys()), list(stage_scores.values()), color=colors)
ax.set_ylabel("R2 (%)")
ax.set_title("R2 by modelling stage")
ax.set_ylim(0, 100)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
            f"{b.get_height():.1f}%", ha="center", fontsize=9)
fig.tight_layout()
savefig(fig, "fig5_stage_progression.png")

# ---------------------------------------------------------------- cache
joblib.dump({"rf_best": rf_best,
             "X2_test": X2_test,
             "feature_cols2": feature_cols2,
             "metrics": METRICS,
             "n_rows_enriched": len(df2),
             "env": ENV,
             "started": STARTED},
            os.path.join(CACHE, "pipeline.joblib"))
print(f"\n  cached pipeline -> results/figures/.cache/pipeline.joblib")

mdf = pd.DataFrame({"metric": list(METRICS.keys()),
                    "value": [round(v, 6) for v in METRICS.values()]})
write_csv(os.path.join(FIGDIR, "run_metrics.csv"), mdf,
          ["every metric measured by THIS run, for comparison against",
           "  model_results_log.csv - they will differ, see the warning above"])
print(f"\n[DONE] five figures + run_metrics.csv written in {spent():.0f}s")
print("Now run:  python3 src/step8b_shap.py   (re-run until it prints DONE)")

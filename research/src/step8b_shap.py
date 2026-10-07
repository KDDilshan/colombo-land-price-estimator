"""Step 8b - SHAP TreeExplainer on the tuned Colombo hazard model.

Loads the model and held-out test set cached by `src/step8_shap_and_figures.py`
and explains every test row. SHAP over a 300-tree depth-20 forest exceeds the
execution window available here, so the matrix is computed in row chunks and
checkpointed to results/figures/.cache/shap_values.npy. RE-RUN UNTIL IT PRINTS
DONE. Chunking is statistically inert: SHAP values are per-row and independent
of the order in which rows are explained. The full test set is explained; no
subsampling is used.

Read the provenance warning in step8_shap_and_figures.py before citing any
number produced here - library versions differ from the notebook run.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "results", "figures")
CACHE = os.path.join(FIGDIR, ".cache")
BUDGET = float(os.environ.get("BUDGET", "150"))
CHUNK = int(os.environ.get("CHUNK", "150"))
_T0 = time.time()
DPI = 300

HAZARD_5 = ["water_occurrence_pct", "dist_to_stream_km", "dist_to_kelani_km",
            "hand_m", "sar_flood_open_land"]
RF_ENR_BEST = dict(n_estimators=300, min_samples_split=4, min_samples_leaf=1,
                   max_features=None, max_depth=20)

def spent(): return time.time() - _T0
def out_of_time(): return spent() > BUDGET

cache = joblib.load(os.path.join(CACHE, "pipeline.joblib"))
rf_best = cache["rf_best"]; X2_test = cache["X2_test"]
feature_cols2 = cache["feature_cols2"]; METRICS = cache["metrics"]
ENV = cache["env"]; STARTED = dt.datetime.now()
N_ROWS_ENR = cache["n_rows_enriched"]

def header(extra):
    lines = [f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step8b_shap.py",
             "# dataset=final_dataset_hazard.csv (7034 rows x 37 cols)",
             f"# env {ENV}",
             "# WARNING library versions differ from the notebook run; values are NOT",
             "#         bit-identical to model_results_log.csv. Do not mix the two.",
             "# hyperparameters=notebook best_params_ instantiated directly, not re-searched"]
    return "\n".join(lines + [f"# {e}" for e in extra]) + "\n"

def write_csv(path, df, extra):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header(extra)); df.to_csv(fh, index=False)
    print(f"  wrote results/figures/{os.path.basename(path)}")

def savefig(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  wrote results/figures/{name}")

df2_len = N_ROWS_ENR
# ---------------------------------------------------------------- SHAP
print("[E] SHAP TreeExplainer on the tuned enriched RandomForest")
SV_PATH = os.path.join(CACHE, "shap_values.npy")
N_TEST = len(X2_test)

if os.path.exists(SV_PATH):
    sv = np.load(SV_PATH)
    if sv.shape != (N_TEST, len(feature_cols2)):
        print("  cached SHAP matrix has the wrong shape; discarding")
        sv = np.full((N_TEST, len(feature_cols2)), np.nan)
else:
    sv = np.full((N_TEST, len(feature_cols2)), np.nan)

done_mask = ~np.isnan(sv[:, 0])
print(f"  {int(done_mask.sum())} / {N_TEST} test rows already explained")

if not done_mask.all():
    explainer = shap.TreeExplainer(rf_best)
    todo = np.where(~done_mask)[0]
    for start in range(0, len(todo), CHUNK):
        idx = todo[start:start + CHUNK]
        t_chunk = time.time()
        sv[idx, :] = np.asarray(
            explainer.shap_values(X2_test.iloc[idx], check_additivity=False))
        np.save(SV_PATH, sv)
        done = int((~np.isnan(sv[:, 0])).sum())
        print(f"  explained {done}/{N_TEST} rows "
              f"(+{len(idx)} in {time.time() - t_chunk:.0f}s, {spent():.0f}s spent)")
        if out_of_time() and done < N_TEST:
            print(f"\n  TIME BUDGET SPENT - {done}/{N_TEST} rows done, "
                  f"progress saved. RE-RUN THIS SCRIPT TO CONTINUE.")
            sys.exit(3)

if np.isnan(sv[:, 0]).any():
    print("  SHAP incomplete; re-run to continue")
    sys.exit(3)
print(f"  SHAP matrix complete: {sv.shape}  (test rows x features)")

mean_abs = np.abs(sv).mean(axis=0)
glob = (pd.DataFrame({"feature": feature_cols2, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True))
glob["rank"] = glob.index + 1
glob["pct_of_total"] = glob["mean_abs_shap"] / glob["mean_abs_shap"].sum() * 100

# direction: sign of the correlation between feature value and its SHAP value
dirs = []
for j, f in enumerate(feature_cols2):
    col = X2_test[f].values.astype(float)
    if np.std(col) == 0 or np.std(sv[:, j]) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(col, sv[:, j])[0, 1])
    dirs.append({"feature": f, "corr_value_vs_shap": r,
                 "direction": ("raises price" if r > 0 else
                               "lowers price" if r < 0 else "undetermined"),
                 "mean_abs_shap": float(mean_abs[j])})
direction = (pd.DataFrame(dirs)
             .sort_values("mean_abs_shap", ascending=False)
             .reset_index(drop=True))

haz_idx = [feature_cols2.index(h) for h in HAZARD_5]
haz_vals = pd.DataFrame({f"shap__{h}": sv[:, i] for h, i in zip(HAZARD_5, haz_idx)})
for h, i in zip(HAZARD_5, haz_idx):
    haz_vals[f"value__{h}"] = X2_test[h].values
haz_vals.insert(0, "test_row", np.arange(len(haz_vals)))

note = [f"model=RandomForestRegressor({RF_ENR_BEST}) fitted on the enriched set",
        f"features={len(feature_cols2)} (48 raw + Address_target_enc); elevation_m INCLUDED",
        f"explained={sv.shape[0]} held-out test rows from the random 80/20 split (random_state=42)",
        f"trim=per-Address +/-1.5 SD -> {df2_len} rows; target=log(Price per Perch)",
        f"this run measured: enriched tuned RF R2={METRICS['8.5_rf']:.2f}%, "
        f"ensemble R2={METRICS['8.6_ensemble']:.2f}%, "
        f"spatial mean R2={METRICS['8.7_spatial_mean']:.2f}%",
        "SHAP is computed on the RANDOM-SPLIT model, so Address_target_enc is",
        "  informative here; under spatial CV it is constant and contributes nothing",
        "  (see RESULTS_COMPILED.md 6.5). Read this alongside permutation importance.",
        ]
write_csv(os.path.join(FIGDIR, "shap_global_importance.csv"), glob, note)
write_csv(os.path.join(FIGDIR, "shap_direction.csv"), direction, note)
write_csv(os.path.join(FIGDIR, "shap_hazard_values.csv"), haz_vals, note)

# SHAP global bar - top 20
top = glob.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
ax.set_xlabel("mean |SHAP value|  (impact on log price/perch)")
ax.set_title("SHAP global feature importance - top 20\n"
             "tuned RandomForest, enriched features, random-split test set")
fig.tight_layout()
savefig(fig, "shap_global_bar.png")

# SHAP beeswarm - the 5 modelled hazard variables only
exp_haz = shap.Explanation(
    values=sv[:, haz_idx],
    data=X2_test[HAZARD_5].values,
    feature_names=HAZARD_5,
)
plt.figure()
shap.plots.beeswarm(exp_haz, max_display=5, show=False)
fig = plt.gcf()
fig.set_size_inches(9, 4.5)
plt.title("SHAP beeswarm - the five modelled hazard variables", fontsize=11)
fig.tight_layout()
savefig(fig, "shap_beeswarm_hazard.png")

print(f"\nSHAP top 10 by mean |SHAP|:")
print(glob.head(10).to_string(index=False))
print(f"\nHazard variables:")
print(glob[glob.feature.isin(HAZARD_5 + ['elevation_m'])].to_string(index=False))
print(f"\n[DONE] SHAP complete in {spent():.0f}s this run")

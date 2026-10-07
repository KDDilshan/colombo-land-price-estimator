"""Step 31 - the Chapter 6 model-comparison table, re-run with NDVI added.

NDVI entered the deployed model in step27 (49 -> 50 features), so the
comparison table in the thesis (notebook cells 25-31) has to be on the same
feature set. Everything else is the notebook's:

  * split        random 80/20, random_state = 42 (1,234 test rows)
  * encoding     out-of-fold target encoding of Address (step16 / cell 23)
  * untuned      RF / ExtraTrees / XGBoost / LightGBM, 300 estimators (cell 25)
  * tuned        the best_params_ the notebook's RandomizedSearchCV found
                 (cells 27-29). The search is NOT repeated: adding one variable
                 that changes R2 by 0.03 pp is not a reason to re-tune, and the
                 deployed forest keeps the same hyperparameters (step27).
  * ensemble     unweighted mean of tuned RF, XGB and LGB on the log scale (cell 31)
  * metrics      R2 and RMSE on log price; MAPE after np.exp (cell 25's report2)

Both feature sets are run. The 49-feature rows must reproduce the notebook's
published numbers before the 50-feature rows are trusted.

Output: data/processed/model_comparison_ndvi.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step16_tuned_ablation_repeated as s16          # noqa: E402
import step25_env9_ablation as s25                    # noqa: E402

SEED = 42
PUBLISHED_49 = {"RandomForest": 89.09, "ExtraTrees": 88.57, "XGBoost": 88.45,
                "LightGBM": 89.57, "RandomForest (tuned)": 89.43,
                "XGBoost (tuned)": 89.68, "LightGBM (tuned)": 89.16,
                "Ensemble RF+XGB+LGB": 90.23}


def models():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, verbosity=-1),
        "RandomForest (tuned)": RandomForestRegressor(
            n_estimators=500, min_samples_split=6, min_samples_leaf=2, max_features=None,
            max_depth=30, random_state=SEED, n_jobs=-1),
        "XGBoost (tuned)": XGBRegressor(
            subsample=1.0, reg_lambda=2.0, n_estimators=500, max_depth=8, learning_rate=0.03,
            colsample_bytree=0.6, random_state=SEED, n_jobs=-1, verbosity=0),
        "LightGBM (tuned)": LGBMRegressor(
            subsample=0.8, num_leaves=63, n_estimators=300, max_depth=-1, learning_rate=0.03,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1),
    }


def score(y, p):
    return dict(r2_pct=r2_score(y, p) * 100,
                rmse_log=float(np.sqrt(mean_squared_error(y, p))),
                mape_pct=mean_absolute_percentage_error(np.exp(y), np.exp(p)) * 100)


def main():
    X, y, groups, _, prov = s25.build_matrix()
    assert prov["rows_used"] == 6167
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=SEED)
    oof, enc = s16.target_encode_oof(groups.iloc[itr], y.iloc[itr], groups.iloc[ite], SEED)
    rows = []
    for fs, extra in (("49 (before NDVI)", []), ("50 (with NDVI)", ["ndvi"])):
        cols = [c for c in X.columns if c not in s25.NEW or c in extra]
        Xtr, Xte = X.iloc[itr][cols].copy(), X.iloc[ite][cols].copy()
        Xtr["Address_target_enc"] = oof.values
        Xte["Address_target_enc"] = enc.values
        preds = {}
        for name, m in models().items():
            m.fit(Xtr, y.iloc[itr])
            preds[name] = m.predict(Xte)
            rows.append(dict(feature_set=fs, model=name, **score(y.iloc[ite], preds[name])))
            print(f"{fs:18s} {name:22s} R2 {rows[-1]['r2_pct']:6.2f}", flush=True)
        ens = (preds["RandomForest (tuned)"] + preds["XGBoost (tuned)"] + preds["LightGBM (tuned)"]) / 3
        rows.append(dict(feature_set=fs, model="Ensemble RF+XGB+LGB", **score(y.iloc[ite], ens)))
        print(f"{fs:18s} {'Ensemble RF+XGB+LGB':22s} R2 {rows[-1]['r2_pct']:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    old = df[df.feature_set.str.startswith("49")].set_index("model").r2_pct.round(2)
    diff = {m: round(old[m] - v, 2) for m, v in PUBLISHED_49.items()}
    print("\n49-feature rows minus the notebook's published R2:", diff)
    df.to_csv(os.path.join(ROOT, "data", "processed", "model_comparison_ndvi.csv"),
              index=False, float_format="%.4f")
    print(df.pivot(index="model", columns="feature_set", values="r2_pct").round(2).to_string())


if __name__ == "__main__":
    main()

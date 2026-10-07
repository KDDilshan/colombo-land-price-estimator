"""Step 27 - rebuild the deployed model with NDVI added to the environmental
and spatial block (six existing variables + ndvi, 50 features).

NDVI is the one new variable that passed the Appendix F screening (step24).
It is added because it passed, not because it raises R2 - step25 found no
detectable change under either protocol.

PROCEDURE = notebook cell 48, reproduced exactly
  * model      RandomForestRegressor(random_state=42, deploy hyperparameters)
               fitted on the random 80/20 training partition (seed 42), with
               the out-of-fold target encoding - step25.run_random's fit.
               Before writing anything, the same procedure WITHOUT ndvi is
               checked against the current deploy/model.pkl: predictions on
               the held-out rows must agree to 1e-9.
  * encoding   smoothed Address mean over all 6,167 rows (smoothing 10)
  * lookup     per-division median of every feature, plus listing_count
  * fallback   global mean log price

Metrics published by predictor.py are recomputed for the new model:
  R2_KNOWN, MEDIAN_APE, MEAN_APE        random split, seed 42
  R2_UNSEEN, *_UNSEEN                   GroupKFold(5) by division, seed 42
                                        (step15.run_spatial, deterministic)

The previous artefacts are copied to deploy/_backup_env6_<date>/ first.

Outputs
  model test/deploy/{model.pkl, feature_order.json, division_values.csv,
                     address_encoding.csv, fallback.json}
  data/processed/deploy_ndvi_metrics.json
"""

import datetime as dt
import json
import os
import re
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step15_spatial_tuned_factorial as s15          # noqa: E402
import step16_tuned_ablation_repeated as s16          # noqa: E402
import step25_env9_ablation as s25                    # noqa: E402

DEPLOY = os.path.join(ROOT, "model test", "deploy")
PROC = os.path.join(ROOT, "data", "processed")
ADD = ["ndvi"]
SEED = 42
FILES = ["model.pkl", "feature_order.json", "division_values.csv",
         "address_encoding.csv", "fallback.json"]


def fit(X, y, groups, cols):
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=SEED)
    oof, enc = s16.target_encode_oof(groups.iloc[itr], y.iloc[itr], groups.iloc[ite], SEED)
    Xtr, Xte = X.iloc[itr][cols].copy(), X.iloc[ite][cols].copy()
    Xtr["Address_target_enc"] = oof.values
    Xte["Address_target_enc"] = enc.values
    m = RandomForestRegressor(random_state=SEED, **s16.TUNED).fit(Xtr, y.iloc[itr])
    return m, Xte, y.iloc[ite]


def main():
    screen = json.load(open(os.path.join(PROC, "env9_screening.json")))
    assert all(a in screen["passing_new"] for a in ADD), "ndvi did not pass screening"
    X, y, groups, meta, prov = s25.build_matrix()
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187
    base_cols = [c for c in X.columns if c not in s25.NEW]
    new_cols = [c for c in X.columns if c not in s25.NEW or c in ADD]

    # ----------------------------------------- 1. the procedure reproduces deploy
    old = joblib.load(os.path.join(DEPLOY, "model.pkl"))
    old_order = json.load(open(os.path.join(DEPLOY, "feature_order.json")))
    if len(old_order) == 49:
        m0, Xte0, _ = fit(X, y, groups, base_cols)
        assert list(Xte0.columns) == old_order
        diff = float(np.abs(old.predict(Xte0) - m0.predict(Xte0)).max())
        print(f"rebuild check vs current model.pkl: max |diff| = {diff:.2e}")
        assert diff < 1e-9
        del m0
    else:
        print(f"current deploy has {len(old_order)} features - rebuild check skipped")

    # ---------------------------------------------------------- 2. new model
    m, Xte, yte = fit(X, y, groups, new_cols)
    rnd = s15.score(yte, m.predict(Xte))
    spa, folds = s15.run_spatial(X[new_cols], y, groups, s16.TUNED, SEED)
    print(f"random split R2 {rnd['r2_pct']:.3f}%  median APE {rnd['medape_pct']:.2f}%  "
          f"mean APE {rnd['mape_pct']:.2f}%")
    print(f"GroupKFold  R2 {spa['r2_pct']:.3f}%  folds "
          f"{[round(f['r2_pct'], 2) for f in folds]}  median APE {spa['medape_pct']:.2f}%  "
          f"mean APE {spa['mape_pct']:.2f}%")

    # ------------------------------------------------------------ 3. lookups
    df = meta.assign(y=y.values)
    gm = float(y.mean())
    st = df.groupby("Address")["y"].agg(["mean", "count"])
    smooth = (st["mean"] * st["count"] + gm * 10) / (st["count"] + 10)
    enc = smooth.rename("Address_target_enc").reset_index()
    div_vals = (pd.concat([meta[["Address"]], X[new_cols]], axis=1)
                .groupby("Address").median())
    div_vals.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in div_vals.columns]
    div_vals["listing_count"] = meta.groupby("Address").size()

    old_div = pd.read_csv(os.path.join(DEPLOY, "division_values.csv"), index_col=0)
    shared = [c for c in old_div.columns if c in div_vals.columns]
    dev = float((old_div[shared] - div_vals.loc[old_div.index, shared]).abs().max().max())
    old_enc = pd.read_csv(os.path.join(DEPLOY, "address_encoding.csv")).set_index("Address")
    dev_enc = float((old_enc.Address_target_enc - enc.set_index("Address")
                     .loc[old_enc.index].Address_target_enc).abs().max())
    print(f"unchanged columns vs current lookup: max |diff| {dev:.2e}; encoding {dev_enc:.2e}")
    assert dev < 1e-9 and dev_enc < 1e-9

    # ------------------------------------------------------------- 4. write
    bak = os.path.join(DEPLOY, f"_backup_env6_{dt.date.today():%Y%m%d}")
    if not os.path.exists(bak):
        os.makedirs(bak)
        for f in FILES:
            shutil.copy2(os.path.join(DEPLOY, f), bak)
        print(f"backed up previous artefacts to {bak}")
    joblib.dump(m, os.path.join(DEPLOY, "model.pkl"))
    json.dump(list(Xte.columns), open(os.path.join(DEPLOY, "feature_order.json"), "w"), indent=1)
    enc.to_csv(os.path.join(DEPLOY, "address_encoding.csv"), index=False)
    div_vals.to_csv(os.path.join(DEPLOY, "division_values.csv"))
    json.dump({"global_mean_log_price": gm}, open(os.path.join(DEPLOY, "fallback.json"), "w"))

    out = dict(generated=dt.datetime.now().isoformat(timespec="seconds"),
               added=ADD, n_features=len(Xte.columns),
               climate_source=screen["climate_source"],
               r2_known=round(rnd["r2_pct"], 2), median_ape=round(rnd["medape_pct"], 1),
               mean_ape=round(rnd["mape_pct"], 1), mae_rs=round(rnd["mae_rs"]),
               r2_unseen=round(spa["r2_pct"], 2),
               r2_unseen_folds=[round(f["r2_pct"], 2) for f in folds],
               median_ape_unseen=round(spa["medape_pct"], 1),
               mean_ape_unseen=round(spa["mape_pct"], 1))
    json.dump(out, open(os.path.join(PROC, "deploy_ndvi_metrics.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

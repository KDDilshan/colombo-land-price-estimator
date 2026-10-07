"""Step 33 - serve the thesis main model (tuned XGBoost) in the web app instead of
the tuned Random Forest (user decision, 2026-09-28).

PROCEDURE = step27's fit with the estimator swapped
  * model      XGBRegressor(random_state=42, step32.XGB), fitted on the random 80/20
               training partition (seed 42) with the out-of-fold target encoding,
               exactly as step32 fits the thesis headline model. Before writing, its
               held-out metrics must reproduce results/xgb_main/metrics.json.
  * lookups    division_values.csv, address_encoding.csv and fallback.json do not
               depend on the estimator; they are recomputed and must match the
               deployed files, which are therefore left as they are.
  * feature order must equal the deployed feature_order.json (50 features).

The previous artefacts are copied to deploy/_backup_rf_<date>/ first.

Outputs
  model test/deploy/model.pkl
  data/processed/deploy_xgb_metrics.json
"""

import datetime as dt
import json
import os
import shutil
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step25_env9_ablation as s25                    # noqa: E402
import step32_xgb_main_model as s32                   # noqa: E402

DEPLOY = os.path.join(ROOT, "model test", "deploy")
PROC = os.path.join(ROOT, "data", "processed")
SEED = 42
FILES = ["model.pkl", "feature_order.json", "division_values.csv",
         "address_encoding.csv", "fallback.json"]


def main():
    X, y, groups, meta, prov = s25.build_matrix()
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187
    cols = [c for c in X.columns if c not in s25.EXISTING + s25.NEW or c in s32.ENV7]

    Xtr, Xte, ytr, yte, _ = s32.split(X, y, groups, cols, SEED)
    order = json.load(open(os.path.join(DEPLOY, "feature_order.json")))
    assert list(Xte.columns) == order, "feature order differs from the deployed one"

    m = s32.fit(Xtr, ytr, SEED)
    got = s32.metrics(yte, m.predict(Xte))
    ref = json.load(open(os.path.join(ROOT, "results", "xgb_main", "metrics.json")))["heldout"]
    for k in ("r2_pct", "medape_pct", "mape_pct"):
        assert abs(got[k] - ref[k]) < 1e-6, (k, got[k], ref[k])
    print({k: round(v, 4) for k, v in got.items()})

    bak = os.path.join(DEPLOY, f"_backup_rf_{dt.date.today():%Y%m%d}")
    if not os.path.exists(bak):
        os.makedirs(bak)
        for f in FILES:
            shutil.copy2(os.path.join(DEPLOY, f), bak)
        print(f"backed up the Random Forest artefacts to {bak}")
    joblib.dump(m, os.path.join(DEPLOY, "model.pkl"))

    # the served file must give the same predictions after a reload
    back = joblib.load(os.path.join(DEPLOY, "model.pkl"))
    assert float(np.abs(back.predict(Xte) - m.predict(Xte)).max()) < 1e-9

    out = dict(generated=dt.datetime.now().isoformat(timespec="seconds"),
               model="XGBRegressor (tuned)", params=s32.XGB, n_features=len(order),
               r2_known=round(got["r2_pct"], 2), median_ape=round(got["medape_pct"], 1),
               mean_ape=round(got["mape_pct"], 1), mae_rs=round(got["mae_rs"]))
    json.dump(out, open(os.path.join(PROC, "deploy_xgb_metrics.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

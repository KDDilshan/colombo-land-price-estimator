"""
step18_spatial_partition_distribution.py
----------------------------------------
How much does the spatially blocked R2 depend on which divisions land in which
fold?

step7 could not ask this. GroupKFold takes no random_state, so the thesis had
only one division-to-fold assignment and reported a standard deviation that
measured forest stochasticity alone. step16 showed the answer matters: over
five randomised partitions the baseline moved by more than 10 points, and the
hazard result that had met the study's decision rule on the fixed partition no
longer met it.

Five replicates is a thin basis for a variance estimate, so this script runs
the deployed configuration over 25 independent size-balanced partitions and
reports the resulting distribution. Nothing else changes: same 6,167 rows, same
187 divisions, same 49 features, same encoder refit inside every fold.

The deterministic GroupKFold partition is evaluated alongside them so its
position within the distribution can be stated rather than assumed.

Outputs
  data/processed/spatial_partition_distribution.csv
  data/processed/spatial_partition_distribution.json
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

_spec = importlib.util.spec_from_file_location(
    "step16", os.path.join(ROOT, "src", "step16_tuned_ablation_repeated.py"))
s16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s16)

N_PARTITIONS = 25
TUNED = s16.TUNED


def evaluate(X, y, groups, folds, seed):
    scores = []
    for tr, va in folds:
        ytr, yva = y.iloc[tr], y.iloc[va]
        oof, enc = s16.target_encode_oof(groups.iloc[tr], ytr, groups.iloc[va], seed)
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        Xtr["Address_target_enc"] = oof.values
        Xva["Address_target_enc"] = enc.values
        m = RandomForestRegressor(random_state=seed, **TUNED).fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)) * 100)
    return np.array(scores)


def main():
    started = dt.datetime.now()
    X, y, groups, prov = s16.build_matrix()
    print(f"rows_used={prov['rows_used']} divisions={prov['divisions']}", flush=True)
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187

    rows = []
    for i in range(1, N_PARTITIONS + 1):
        folds = list(s16.shuffled_group_folds(groups, i))
        s = evaluate(X, y, groups, folds, i)
        rows.append({"partition": i, "kind": "randomised",
                     **{f"fold{j+1}_r2": round(v, 4) for j, v in enumerate(s)},
                     "mean_r2": round(s.mean(), 4),
                     "min_fold": round(s.min(), 4), "max_fold": round(s.max(), 4)})
        print(f"  partition {i:>2}  mean {s.mean():6.2f}  "
              f"folds {np.round(s, 1)}", flush=True)

    gk = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
    sg = evaluate(X, y, groups, gk, 42)
    rows.append({"partition": 0, "kind": "deterministic_GroupKFold",
                 **{f"fold{j+1}_r2": round(v, 4) for j, v in enumerate(sg)},
                 "mean_r2": round(sg.mean(), 4),
                 "min_fold": round(sg.min(), 4), "max_fold": round(sg.max(), 4)})
    print(f"\n  GroupKFold (deterministic, seed 42)  mean {sg.mean():6.2f}  "
          f"folds {np.round(sg, 1)}", flush=True)

    df = pd.DataFrame(rows)
    rnd = df[df.kind == "randomised"]["mean_r2"]
    pctile = float((rnd < sg.mean()).mean() * 100)

    head = dict(
        generated=str(started), **prov, n_features=49,
        model=("RandomForestRegressor(n_estimators=500, max_depth=30, "
               "max_features=None, min_samples_split=6, min_samples_leaf=2)"),
        n_randomised_partitions=N_PARTITIONS,
        randomised_mean_r2=round(float(rnd.mean()), 4),
        randomised_sd_r2=round(float(rnd.std(ddof=1)), 4),
        randomised_min_r2=round(float(rnd.min()), 4),
        randomised_max_r2=round(float(rnd.max()), 4),
        randomised_p05_r2=round(float(rnd.quantile(0.05)), 4),
        randomised_p95_r2=round(float(rnd.quantile(0.95)), 4),
        deterministic_groupkfold_r2=round(float(sg.mean()), 4),
        deterministic_percentile_within_randomised=round(pctile, 1),
        worst_single_fold=round(float(df.min_fold.min()), 4),
        best_single_fold=round(float(df.max_fold.max()), 4),
    )

    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by "
            f"src/step18_spatial_partition_distribution.py\n"
            f"# rows_used={prov['rows_used']} divisions={prov['divisions']} features=49\n"
            f"# model=RandomForestRegressor(n_estimators=500, max_depth=30, "
            f"max_features=None, min_samples_split=6, min_samples_leaf=2)\n"
            f"# {N_PARTITIONS} independent size-balanced division-to-fold assignments,\n"
            f"#   plus the deterministic GroupKFold partition as partition 0\n")
    path = os.path.join(PROC, "spatial_partition_distribution.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(meta)
        df.to_csv(fh, index=False)
    with open(os.path.join(PROC, "spatial_partition_distribution.json"), "w",
              encoding="utf-8") as fh:
        json.dump(head, fh, indent=2)

    print("\n===== spatially blocked R2 across partitions =====")
    for k, v in head.items():
        if k.startswith(("randomised", "deterministic", "worst", "best")):
            print(f"  {k:45s} {v}")
    print(f"\nelapsed {dt.datetime.now() - started}")


if __name__ == "__main__":
    main()

"""
step19_ablation_attribution_ladder.py
-------------------------------------
Attributes the change in the hazard-block result to its cause.

The thesis published a hazard-block contribution of -1.00 pp that met the
study's pre-specified decision rule. Re-running under the deployed
configuration (step16, step17) returns a much smaller effect that does not meet
the rule. Three things differ between the published run and the re-run:

  (a) the estimator      untuned 300-tree  ->  deployed tuned 500-tree
  (b) the hazard block   five variables (elevation dropped from the model
                         entirely)  ->  all six retained variables
  (c) the partition      one fixed GroupKFold split  ->  a different randomised
                         split per replicate

Changing three things at once cannot say which mattered. This script walks the
ladder one rung at a time, so each step differs from the one above it in
exactly one respect:

  rung 1  untuned 300, five-variable block, fixed folds   [= step7, reproduced]
  rung 2  tuned  500, five-variable block, fixed folds    [(a) only]
  rung 3  tuned  500, six-variable block,  fixed folds    [(a)+(b)]  = step17
  rung 4  tuned  500, six-variable block,  randomised     [(a)+(b)+(c)] = step16

Rungs 1 and 2 are computed here; rungs 3 and 4 are read from the artefacts the
other two scripts wrote, so the ladder cannot drift from them.

Output
  data/processed/ablation_attribution_ladder.csv
  data/processed/ablation_attribution_ladder.json
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

SEEDS = s16.REPLICATES
ALL6 = s16.HAZARD_COLS
FIVE = [h for h in ALL6 if h != "elevation_m"]      # step7's ablation block
TUNED = s16.TUNED
UNTUNED = dict(n_estimators=300, n_jobs=-1)


def cv_fixed(X, y, groups, cols, kw, seed):
    scores = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups=groups):
        ytr, yva = y.iloc[tr], y.iloc[va]
        oof, enc = s16.target_encode_oof(groups.iloc[tr], ytr, groups.iloc[va], seed)
        Xtr, Xva = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy()
        Xtr["Address_target_enc"] = oof.values
        Xva["Address_target_enc"] = enc.values
        m = RandomForestRegressor(random_state=seed, **kw).fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)) * 100)
    return float(np.mean(scores))


def rung(X, y, groups, kw, block, label):
    """block = the hazard variables treated as the ablation block. Variables
    outside the block are dropped from the model entirely, matching step7's
    treatment of elevation_m."""
    inmodel = [c for c in X.columns if c not in ALL6 or c in block]
    nohaz = [c for c in inmodel if c not in block]
    base_by_seed, nohaz_by_seed = {}, {}
    for seed in SEEDS:
        base_by_seed[seed] = cv_fixed(X, y, groups, inmodel, kw, seed)
        nohaz_by_seed[seed] = cv_fixed(X, y, groups, nohaz, kw, seed)
        print(f"  {label:44s} seed {seed:>4}  base {base_by_seed[seed]:6.2f}  "
              f"no_hazard {nohaz_by_seed[seed]:6.2f}", flush=True)
    b = pd.Series(base_by_seed)
    n = pd.Series(nohaz_by_seed)
    d = n - b                      # negative = removing hazard hurts
    same = int((np.sign(d) == np.sign(d.mean())).sum())
    sd = float(d.std(ddof=1))
    return dict(rung=label, n_features_baseline=len(inmodel) + 1,
                block_size=len(block), n_estimators=kw["n_estimators"],
                tuned=("max_depth" in kw), partition="fixed GroupKFold",
                baseline_mean_r2=round(float(b.mean()), 4),
                no_hazard_mean_r2=round(float(n.mean()), 4),
                delta_pp=round(float(d.mean()), 4),
                delta_sd_pp=round(sd, 4), two_sd_pp=round(2 * sd, 4),
                replicates_same_sign=same, n_replicates=len(SEEDS),
                meets_decision_rule=bool(abs(d.mean()) > 2 * sd
                                         and same == len(SEEDS)))


def main():
    started = dt.datetime.now()
    X, y, groups, prov = s16.build_matrix()
    print(f"rows_used={prov['rows_used']} divisions={prov['divisions']}", flush=True)
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187

    rows = [
        rung(X, y, groups, UNTUNED, FIVE,
             "1: untuned300 / 5-var block / fixed folds"),
        rung(X, y, groups, TUNED, FIVE,
             "2: tuned500 / 5-var block / fixed folds"),
    ]

    # rungs 3 and 4 read from the artefacts that produced them
    r3 = json.load(open(os.path.join(PROC, "tuned_ablation_fixedfolds.json"),
                        encoding="utf-8"))
    r4 = json.load(open(os.path.join(PROC, "tuned_ablation_repeated.json"),
                        encoding="utf-8"))
    rows.append(dict(rung="3: tuned500 / 6-var block / fixed folds",
                     n_features_baseline=49, block_size=6, n_estimators=500,
                     tuned=True, partition="fixed GroupKFold",
                     baseline_mean_r2=r3["baseline_all6_mean_r2"],
                     no_hazard_mean_r2=r3["no_hazard_mean_r2"],
                     delta_pp=r3["hazard_block_delta_pp"],
                     delta_sd_pp=r3["hazard_block_delta_sd_pp"],
                     two_sd_pp=round(2 * r3["hazard_block_delta_sd_pp"], 4),
                     replicates_same_sign=None, n_replicates=5,
                     meets_decision_rule=r3["hazard_block_meets_rule"]))
    rows.append(dict(rung="4: tuned500 / 6-var block / randomised partitions",
                     n_features_baseline=49, block_size=6, n_estimators=500,
                     tuned=True, partition="randomised, per replicate",
                     baseline_mean_r2=r4["baseline_all6_mean_r2"],
                     no_hazard_mean_r2=r4["no_hazard_mean_r2"],
                     delta_pp=r4["hazard_block_delta_pp"],
                     delta_sd_pp=r4["hazard_block_delta_sd_pp"],
                     two_sd_pp=round(2 * r4["hazard_block_delta_sd_pp"], 4),
                     replicates_same_sign=None, n_replicates=5,
                     meets_decision_rule=r4["hazard_block_meets_rule"]))

    df = pd.DataFrame(rows)
    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by "
            f"src/step19_ablation_attribution_ladder.py\n"
            f"# Each rung differs from the one above it in exactly one respect,\n"
            f"#   so the change in delta_pp between adjacent rungs is attributable\n"
            f"#   to that one change. Rungs 1-2 computed here; rungs 3-4 read from\n"
            f"#   tuned_ablation_fixedfolds.json and tuned_ablation_repeated.json.\n"
            f"# rows_used={prov['rows_used']} divisions={prov['divisions']}\n"
            f"# seeds/replicates={'|'.join(map(str, SEEDS))}\n")
    path = os.path.join(PROC, "ablation_attribution_ladder.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(meta)
        df.to_csv(fh, index=False)
    with open(os.path.join(PROC, "ablation_attribution_ladder.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(generated=str(started), **prov, rungs=rows), fh, indent=2)

    print("\n===== attribution ladder =====")
    print(df[["rung", "baseline_mean_r2", "no_hazard_mean_r2", "delta_pp",
              "delta_sd_pp", "two_sd_pp", "meets_decision_rule"]].to_string(index=False))
    print(f"\nelapsed {dt.datetime.now() - started}")


if __name__ == "__main__":
    main()

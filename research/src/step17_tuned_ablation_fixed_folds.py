"""
step17_tuned_ablation_fixed_folds.py
------------------------------------
Control run for src/step16_tuned_ablation_repeated.py.

step16 changed three things at once relative to the published ablation of
step7: the estimator (untuned 300-tree -> deployed tuned 500-tree), the hazard
block (five variables -> all six retained), and the partition (one fixed
GroupKFold split -> a different randomised split per replicate). If the hazard
result moves, three explanations are available and the run cannot say which.

This script holds the estimator and the hazard block at step16's settings and
reverts only the partition to the deterministic GroupKFold that step7 used.
Comparing the two therefore attributes any change in the hazard result to the
partition alone, which is the question at issue: whether the published effect
was a property of the hazard variables or of one particular division-to-fold
assignment.

Everything else is imported directly from step16 so the two runs cannot drift
apart.

Outputs
  data/processed/tuned_ablation_fixedfolds_runs.csv
  data/processed/tuned_ablation_fixedfolds_summary.csv
  data/processed/tuned_ablation_fixedfolds.json
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
HAZARD_COLS = s16.HAZARD_COLS
TUNED = s16.TUNED
N_SPLITS = s16.N_SPLITS


def run_cv_fixed(X, y, groups, cols, seed):
    """Identical to step16.run_cv except the splitter: deterministic GroupKFold,
    so the partition is the same for every seed and only the forest and the
    inner encoder split vary."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = []
    for tr, va in gkf.split(X, y, groups=groups):
        ytr, yva = y.iloc[tr], y.iloc[va]
        oof, enc = s16.target_encode_oof(groups.iloc[tr], ytr, groups.iloc[va], seed)
        Xtr, Xva = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy()
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

    allcols = list(X.columns)
    configs = {"baseline_all6_hazard": allcols,
               "no_hazard": [c for c in allcols if c not in HAZARD_COLS]}
    for h in HAZARD_COLS:
        configs[f"drop_{h}"] = [c for c in allcols if c != h]

    rows = []
    for cfg, cols in configs.items():
        for seed in SEEDS:
            s = run_cv_fixed(X, y, groups, cols, seed)
            rows.append({"config": cfg, "seed": seed, "n_features": len(cols) + 1,
                         **{f"fold{i+1}_r2": round(v, 4) for i, v in enumerate(s)},
                         "mean_r2": round(s.mean(), 4),
                         "sd_across_folds": round(s.std(ddof=1), 4)})
            print(f"  {cfg:26s} seed {seed:>4}  mean {s.mean():6.2f}  "
                  f"folds {np.round(s, 2)}", flush=True)

    runs = pd.DataFrame(rows)
    base_by_seed = (runs[runs.config == "baseline_all6_hazard"]
                    .set_index("seed")["mean_r2"])

    summ = []
    for cfg in configs:
        sub = runs[runs.config == cfg].set_index("seed")["mean_r2"]
        if cfg == "baseline_all6_hazard":
            delta, dsd, same, clears = sub * 0, 0.0, "", ""
        else:
            delta = sub - base_by_seed
            dsd = float(delta.std(ddof=1))
            same = int((np.sign(delta) == np.sign(delta.mean())).sum())
            clears = bool(abs(delta.mean()) > 2 * dsd and same == len(SEEDS))
        summ.append({"config": cfg,
                     "n_features": int(runs[runs.config == cfg].n_features.iloc[0]),
                     "mean_r2": round(float(sub.mean()), 4),
                     "sd_across_seeds": round(float(sub.std(ddof=1)), 4),
                     "delta_vs_baseline_pp": round(float(delta.mean()), 4),
                     "delta_sd_pp": round(dsd, 4),
                     "two_sd_pp": round(2 * dsd, 4),
                     "seeds_same_sign": same,
                     "meets_decision_rule": clears})
    summary = pd.DataFrame(summ)
    s = summary.set_index("config")

    head = dict(generated=str(started), **prov, n_features=49, seeds=SEEDS,
                partition="deterministic GroupKFold(5) by Address - identical "
                          "across seeds",
                model="RandomForestRegressor(n_estimators=500, max_depth=30, "
                      "max_features=None, min_samples_split=6, min_samples_leaf=2)",
                baseline_all6_mean_r2=float(s.loc["baseline_all6_hazard", "mean_r2"]),
                baseline_sd_across_seeds=float(
                    s.loc["baseline_all6_hazard", "sd_across_seeds"]),
                no_hazard_mean_r2=float(s.loc["no_hazard", "mean_r2"]),
                hazard_block_delta_pp=float(s.loc["no_hazard", "delta_vs_baseline_pp"]),
                hazard_block_delta_sd_pp=float(s.loc["no_hazard", "delta_sd_pp"]),
                hazard_block_meets_rule=bool(s.loc["no_hazard", "meets_decision_rule"]))

    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by "
            f"src/step17_tuned_ablation_fixed_folds.py\n"
            f"# CONTROL for step16: same estimator, same six-variable hazard block,\n"
            f"#   but the deterministic GroupKFold partition of step7 instead of a\n"
            f"#   randomised one. Differences against step16 are due to the partition.\n"
            f"# rows_used={prov['rows_used']} divisions={prov['divisions']} features=49\n"
            f"# model=RandomForestRegressor(n_estimators=500, max_depth=30, "
            f"max_features=None, min_samples_split=6, min_samples_leaf=2)\n"
            f"# cv=GroupKFold(n_splits=5) by Address - DETERMINISTIC, identical across seeds\n"
            f"# seeds={'|'.join(map(str, SEEDS))} (vary the forest and encoder only)\n")
    for path, frame in ((os.path.join(PROC, "tuned_ablation_fixedfolds_runs.csv"), runs),
                        (os.path.join(PROC, "tuned_ablation_fixedfolds_summary.csv"), summary)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(meta)
            frame.to_csv(fh, index=False)
    with open(os.path.join(PROC, "tuned_ablation_fixedfolds.json"), "w",
              encoding="utf-8") as fh:
        json.dump(head, fh, indent=2)

    print("\n===== ablation, deployed configuration, FIXED GroupKFold partition =====")
    print(summary.to_string(index=False))
    print(f"\nelapsed {dt.datetime.now() - started}")


if __name__ == "__main__":
    main()

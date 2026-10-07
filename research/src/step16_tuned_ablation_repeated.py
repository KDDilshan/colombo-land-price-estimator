"""
step16_tuned_ablation_repeated.py
---------------------------------
Two corrections to the hazard ablation of src/step7_hazard_ablation.py, both
raised in the thesis audit.

1. WRONG MODEL. step7 used an untuned 300-tree Random Forest while the model
   the thesis deploys and reports is the tuned 500-tree configuration. The
   ablation therefore did not test the deployed specification. This script
   uses deploy/model.pkl's exact hyperparameters.

2. FOLD ASSIGNMENT WAS NEVER VARIED. GroupKFold has no random_state; it
   partitions groups deterministically by size. step7 ran five seeds, but all
   five shared one partition, so the reported spread measured forest
   stochasticity only and understated the true uncertainty. The thesis had to
   disclose this as a limitation. Here the division-to-fold assignment is
   randomised independently in each replicate, so the spread being reported
   now includes partition variance and the limitation is removed rather than
   disclosed.

A third change removes a terminological contradiction rather than a defect.
step7 ablated five hazard variables, excluding elevation_m, while the model
retains six. Readers could not tell whether the block was five or six. The
ablation here covers all SIX retained hazard/terrain variables, so the
retained block and the ablated block are the same object.

Design
------
* Same cleaned frame as the deployed model: 7,034 -> price filter -> per-
  division +/-1.5 SD trim -> 6,167 rows across 187 divisions, 49 features.
* Grouped 5-fold CV by division. In each replicate the 187 divisions are
  shuffled and dealt into five folds, so every replicate is a different
  spatial partition. Replicate r uses seed r for the partition, the forest
  and the inner encoder KFold alike.
* Target encoder refit inside every fold on training rows only; divisions
  unseen in a fold receive the training global mean.
* Deltas are paired within replicate, so the partition-level shift shared by
  a config and its baseline cancels. A delta is judged detectable only if its
  absolute mean exceeds two standard deviations of its own paired
  distribution AND its sign is consistent across replicates - the same
  pre-specified rule step7 applied, now evaluated against a variance estimate
  that includes the partition.

Outputs
  data/processed/tuned_ablation_repeated_runs.csv
  data/processed/tuned_ablation_repeated_summary.csv
  data/processed/tuned_ablation_repeated.json
  data/processed/tuned_ablation_repeated_report.md
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
DATA = os.path.join(PROC, "final_dataset_hazard.csv")

REPLICATES = [42, 1, 7, 123, 2024]
K_TRIM = 1.5
N_SPLITS = 5
SMOOTHING = 10

HAZARD_COLS = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
               "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]

TUNED = dict(n_estimators=500, max_depth=30, max_features=None,
             min_samples_split=6, min_samples_leaf=2, n_jobs=-1)


def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, group="Address", k=K_TRIM):
    g = d.groupby(group)[col]
    mean, sd = g.transform("mean"), g.transform("std")
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], int((~keep).sum())


def target_encode_oof(train_groups, train_target, test_groups, seed,
                      n_splits=5, smoothing=SMOOTHING):
    global_mean = train_target.mean()
    oof = pd.Series(index=train_groups.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
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


def build_matrix():
    df = pd.read_csv(DATA, low_memory=False)
    n_in = len(df)
    lo, hi = df["Price per Perch"].quantile([0.005, 0.995])
    df = df[df["Price per Perch"].between(lo, hi)].copy()
    n_floor = n_in - len(df)
    df, n1 = drop_per_group(df, "Price per Perch")
    df, n2 = drop_per_group(df, "Land_size(Perches)")
    y = np.log(df["Price per Perch"])

    count_cols = [c for c in df.columns if c.startswith("count_")]
    mindist_cols = [c for c in df.columns if c.startswith("min_dist")]
    df["total_amenity_count"] = df[count_cols].sum(axis=1)
    df["min_dist_any_amenity"] = df[mindist_cols].min(axis=1)
    df["school_count_total"] = (df["count_govtschools_A"]
                                + df["count_semigovtschools"]
                                + df["count_intlschools"])
    df["health_count_total"] = (df["count_Govt_Hospitals"]
                                + df["count_Pvt_Hospital"]
                                + df["count_Pvt_Med_Centers"])
    df["finance_count_total"] = (df["count_banks_within_2km"]
                                 + df["count_FinanceCompanies_within_2km"])
    engineered = ["total_amenity_count", "min_dist_any_amenity",
                  "school_count_total", "health_count_total", "finance_count_total"]

    dummies = pd.get_dummies(df["Land_type"], prefix="Land_type").astype(int)
    amenity = [c for c in df.columns
               if c.startswith(("min_dist", "count_")) and c not in engineered]
    base = ["Land_size(Perches)", "Distance from fort"] + amenity + HAZARD_COLS + engineered
    X = pd.concat([df[base], dummies], axis=1)
    X.columns = [clean_name(c) for c in X.columns]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = df["Address"].reset_index(drop=True)
    return X, y, groups, dict(rows_in=n_in, price_filtered=n_floor,
                              trimmed=n1 + n2, rows_used=len(df),
                              divisions=int(groups.nunique()))


def shuffled_group_folds(groups, seed, n_splits=N_SPLITS):
    """Randomised but size-balanced division-to-fold assignment.

    GroupKFold assigns groups to folds deterministically: it walks them in
    descending size order and gives each to the fold holding fewest rows so
    far, which balances fold sizes but fixes the partition. Balancing matters
    here because division sizes span 1 to 518 listings, and a purely random
    assignment can put several large divisions in one fold and leave another
    fold too thin for R2 to mean anything.

    This keeps the balancing and randomises only the order in which groups are
    walked, so fold sizes stay comparable to GroupKFold's while the partition
    genuinely differs from replicate to replicate. Variance across replicates
    therefore includes partition variance, which the deterministic GroupKFold
    used by step7 could not measure.
    """
    sizes = groups.value_counts()
    uniq = np.array(sizes.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    # walk large groups first so the greedy balance has room to work, but break
    # ties -- and they are frequent, since most divisions are small -- at random
    order = sorted(uniq, key=lambda g: (-int(sizes[g]), int(rng.integers(1 << 30))))
    load = np.zeros(n_splits, dtype=int)
    assign = {}
    for g in order:
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += int(sizes[g])
    fold_of_row = groups.map(assign).to_numpy()
    idx = np.arange(len(groups))
    for f in range(n_splits):
        va = idx[fold_of_row == f]
        tr = idx[fold_of_row != f]
        yield tr, va


def run_cv(X, y, groups, cols, seed):
    scores, sizes = [], []
    for tr, va in shuffled_group_folds(groups, seed):
        ytr, yva = y.iloc[tr], y.iloc[va]
        oof, enc = target_encode_oof(groups.iloc[tr], ytr, groups.iloc[va], seed)
        Xtr, Xva = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy()
        Xtr["Address_target_enc"] = oof.values
        Xva["Address_target_enc"] = enc.values
        m = RandomForestRegressor(random_state=seed, **TUNED).fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)) * 100)
        sizes.append((len(tr), len(va), int(groups.iloc[va].nunique())))
    return np.array(scores), sizes


def main():
    started = dt.datetime.now()
    X, y, groups, prov = build_matrix()
    print(f"rows_in={prov['rows_in']} price_filtered={prov['price_filtered']} "
          f"trimmed={prov['trimmed']} rows_used={prov['rows_used']} "
          f"divisions={prov['divisions']}", flush=True)
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187

    allcols = list(X.columns)
    configs = {"baseline_all6_hazard": allcols,
               "no_hazard": [c for c in allcols if c not in HAZARD_COLS]}
    for h in HAZARD_COLS:
        configs[f"drop_{h}"] = [c for c in allcols if c != h]

    rows = []
    for cfg, cols in configs.items():
        for rep in REPLICATES:
            s, sizes = run_cv(X, y, groups, cols, rep)
            rows.append({"config": cfg, "replicate_seed": rep,
                         "n_features": len(cols) + 1,
                         **{f"fold{i+1}_r2": round(v, 4) for i, v in enumerate(s)},
                         "mean_r2": round(s.mean(), 4),
                         "sd_across_folds": round(s.std(ddof=1), 4),
                         "min_fold_r2": round(s.min(), 4),
                         "max_fold_r2": round(s.max(), 4)})
            print(f"  {cfg:26s} rep {rep:>4}  mean {s.mean():6.2f}  "
                  f"folds {np.round(s, 2)}", flush=True)

    runs = pd.DataFrame(rows)
    base_by_rep = (runs[runs.config == "baseline_all6_hazard"]
                   .set_index("replicate_seed")["mean_r2"])

    summ = []
    for cfg in configs:
        sub = runs[runs.config == cfg].set_index("replicate_seed")["mean_r2"]
        if cfg == "baseline_all6_hazard":
            delta = sub * 0
            same, clears, dsd = "", "", 0.0
        else:
            delta = sub - base_by_rep
            dsd = float(delta.std(ddof=1))
            same = int((np.sign(delta) == np.sign(delta.mean())).sum())
            clears = bool(abs(delta.mean()) > 2 * dsd and same == len(REPLICATES))
        summ.append({
            "config": cfg,
            "n_features": int(runs[runs.config == cfg].n_features.iloc[0]),
            "mean_r2": round(float(sub.mean()), 4),
            "sd_across_replicates": round(float(sub.std(ddof=1)), 4),
            "min_replicate": round(float(sub.min()), 4),
            "max_replicate": round(float(sub.max()), 4),
            "delta_vs_baseline_pp": round(float(delta.mean()), 4),
            "delta_sd_pp": round(dsd, 4),
            "two_sd_pp": round(2 * dsd, 4),
            "replicates_same_sign": same,
            "meets_decision_rule": clears,
        })
    summary = pd.DataFrame(summ)

    s = summary.set_index("config")
    base = float(s.loc["baseline_all6_hazard", "mean_r2"])
    base_sd = float(s.loc["baseline_all6_hazard", "sd_across_replicates"])
    nohaz = float(s.loc["no_hazard", "mean_r2"])

    head = dict(
        generated=str(started), **prov, n_features=49, replicates=REPLICATES,
        model="RandomForestRegressor(n_estimators=500, max_depth=30, "
              "max_features=None, min_samples_split=6, min_samples_leaf=2)",
        partition="randomised division-to-fold assignment, 5 folds, "
                  "independent per replicate",
        baseline_all6_mean_r2=round(base, 4),
        baseline_sd_across_partitions=round(base_sd, 4),
        no_hazard_mean_r2=round(nohaz, 4),
        hazard_block_delta_pp=round(float(s.loc["no_hazard", "delta_vs_baseline_pp"]), 4),
        hazard_block_delta_sd_pp=round(float(s.loc["no_hazard", "delta_sd_pp"]), 4),
        hazard_block_meets_rule=bool(s.loc["no_hazard", "meets_decision_rule"]),
        any_single_variable_meets_rule=bool(
            any(bool(s.loc[c, "meets_decision_rule"]) for c in configs
                if c.startswith("drop_"))),
    )

    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by src/step16_tuned_ablation_repeated.py\n"
            f"# dataset={os.path.basename(DATA)} rows_in={prov['rows_in']} "
            f"price_filtered={prov['price_filtered']} trimmed_k={K_TRIM} "
            f"rows_used={prov['rows_used']} divisions={prov['divisions']} features=49\n"
            f"# model=RandomForestRegressor(n_estimators=500, max_depth=30, max_features=None, "
            f"min_samples_split=6, min_samples_leaf=2)  [= deploy/model.pkl]\n"
            f"# cv=5-fold grouped by Address with RANDOMISED division-to-fold assignment\n"
            f"#    (not GroupKFold: the partition itself varies across replicates)\n"
            f"# encoder=oof_target_enc(KFold5, smoothing=10) refit inside every fold\n"
            f"# hazard_set=all 6 retained variables: {'|'.join(HAZARD_COLS)}\n"
            f"# replicates={'|'.join(map(str, REPLICATES))} "
            f"(each varies partition, forest and encoder together)\n")
    for path, frame in ((os.path.join(PROC, "tuned_ablation_repeated_runs.csv"), runs),
                        (os.path.join(PROC, "tuned_ablation_repeated_summary.csv"), summary)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(meta)
            frame.to_csv(fh, index=False)
    with open(os.path.join(PROC, "tuned_ablation_repeated.json"), "w",
              encoding="utf-8") as fh:
        json.dump(head, fh, indent=2)

    md = [
        "# Hazard ablation under the deployed configuration, repeated spatial partitions",
        "",
        f"Generated {started:%Y-%m-%d %H:%M} by `src/step16_tuned_ablation_repeated.py`.",
        "Supersedes `src/step7_hazard_ablation.py` on two counts: the estimator is the",
        "deployed tuned 500-tree Random Forest rather than an untuned 300-tree forest,",
        "and the division-to-fold assignment is randomised per replicate rather than",
        "held fixed, so the reported spread includes partition variance.",
        "",
        "## Specification", "", "| | |", "|---|---|",
        f"| rows / divisions / features | {prov['rows_used']} / {prov['divisions']} / 49 |",
        "| model | `RandomForestRegressor(n_estimators=500, max_depth=30, "
        "max_features=None, min_samples_split=6, min_samples_leaf=2)` |",
        "| validation | 5-fold grouped by division, randomised assignment per replicate |",
        "| target encoding | out-of-fold, KFold(5), smoothing 10, refit inside every fold |",
        f"| hazard block | all six retained variables |",
        f"| replicates | {', '.join(map(str, REPLICATES))} |",
        "",
        "## Results", "",
        "| config | features | mean R² | SD across partitions | range | Δ vs baseline (pp) | 2 SD | same sign | meets rule |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        d = "—" if r.config == "baseline_all6_hazard" else f"{r.delta_vs_baseline_pp:+.2f}"
        t = "—" if r.config == "baseline_all6_hazard" else f"{r.two_sd_pp:.2f}"
        ss = "—" if r.config == "baseline_all6_hazard" else f"{r.replicates_same_sign}/{len(REPLICATES)}"
        mr = "—" if r.config == "baseline_all6_hazard" else ("**yes**" if r.meets_decision_rule else "no")
        md.append(f"| `{r.config}` | {r.n_features} | {r.mean_r2:.2f} | "
                  f"{r.sd_across_replicates:.2f} | {r.min_replicate:.2f}–{r.max_replicate:.2f} | "
                  f"{d} | {t} | {ss} | {mr} |")
    md += ["",
           "## Reading", "",
           f"The baseline varies by {base_sd:.2f} pp SD across partitions. Deltas are paired "
           "within replicate so that this partition-level shift cancels; the decision rule is "
           "the study's pre-specified one (|mean Δ| > 2 SD of its own paired distribution, with "
           "a sign consistent across all replicates). It is a decision rule adopted for this "
           "study, not a conventional significance test, and should not be reported as one.",
           ""]
    with open(os.path.join(PROC, "tuned_ablation_repeated_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print("\n===== ablation, deployed configuration, randomised partitions =====")
    print(summary.to_string(index=False))
    print(f"\nelapsed {dt.datetime.now() - started}")


if __name__ == "__main__":
    main()

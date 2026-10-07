"""Step 7 - reproducible multi-seed leave-one-out ablation of the hazard block.

WHY THIS EXISTS
  `model test/hazard_ablation_results.csv` and
  `model test/permutation_importance_hazard_run.csv` were produced by code that
  is not in the repository. The headline hazard finding therefore could not be
  reproduced, and the disagreement between that file's spatial baseline (54.01)
  and `model_results_log.csv` 8.7_spatial_cv (53.52) could not be diagnosed.
  This script rebuilds the ablation from the pipeline in
  `random_forest_land_price_model.ipynb`, over multiple seeds, and writes its
  own provenance into the output.

PIPELINE - matched cell by cell to the notebook
  cell 21  per-Address outlier trim at k=1.5 on 'Price per Perch' and
           'Land_size(Perches)', then the five engineered aggregates
           (total_amenity_count, min_dist_any_amenity, school_count_total,
           health_count_total, finance_count_total) and Land_type dummies
  cell 23  leak-safe out-of-fold target encoding of Address, KFold(5),
           smoothing=10, refit INSIDE each spatial fold
  cell 33  GroupKFold(n_splits=5) grouped by Address, so every validation
           division is unseen in training;
           RandomForestRegressor(n_estimators=300, n_jobs=-1)

  NOTE ON THE MODEL. `model_results_log.csv` labels stage 8.7_spatial_cv as
  "RF+XGB+LGB avg", but notebook cell 33 fits a RandomForest ONLY. The code is
  the authority; this script uses RandomForest and the mislabel is recorded in
  RESULTS_COMPILED.md.

WHAT THE SEED DOES, AND DOES NOT, VARY
  GroupKFold has no random_state - it partitions groups deterministically by
  group size. **The five folds are therefore identical across seeds.** A seed
  varies only the RandomForest and the inner KFold of the target encoder. The
  spread measured here is model stochasticity, NOT fold-assignment variance, so
  it is a LOWER BOUND on the true run-to-run noise. Stated in the output.

CONFIGURATIONS
  baseline_all5_hazard, no_hazard, and drop-one for each of the five hazard
  variables. `elevation_m` is excluded throughout - it is in the file but not in
  the modelling set (hazard_screening.md 3c).

Outputs
  data/processed/hazard_ablation_multiseed.csv     one row per config x seed
  data/processed/hazard_ablation_summary.csv       one row per config
  data/processed/hazard_ablation_report.md
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
DATA = os.path.join(PROC, "final_dataset_hazard.csv")
OUT_RUNS = os.path.join(PROC, "hazard_ablation_multiseed.csv")
OUT_SUM = os.path.join(PROC, "hazard_ablation_summary.csv")
OUT_MD = os.path.join(PROC, "hazard_ablation_report.md")

SEEDS = [42, 1, 7, 123, 2024]
K_TRIM = 1.5
N_SPLITS = 5
N_ESTIMATORS = 300
SMOOTHING = 10

HAZARD_COLS = ["water_occurrence_pct", "dist_to_stream_km", "dist_to_kelani_km",
               "hand_m", "sar_flood_open_land"]
EXCLUDED_FROM_MODEL = ["elevation_m"]      # hazard_screening.md 3c

# the claims this run is meant to test, from hazard_ablation_results.csv
PRIOR = {
    "baseline_all5_hazard": 54.01, "no_hazard": 49.94,
    "drop_water_occurrence_pct": 51.89, "drop_dist_to_stream_km": 53.71,
    "drop_dist_to_kelani_km": 53.15, "drop_hand_m": 53.01,
    "drop_sar_flood_open_land": 53.53,
}
PRIOR_FOLD1 = {"baseline_all5_hazard": 53.17, "no_hazard": 58.23}
PRIOR_FOLD2 = {"baseline_all5_hazard": 60.78, "no_hazard": 48.29}


def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, group="Address", k=K_TRIM):
    g = d.groupby(group)[col]
    mean, sd = g.transform("mean"), g.transform("std")
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], int((~keep).sum())


def target_encode_oof(train_groups, train_target, test_groups, seed,
                      n_splits=5, smoothing=SMOOTHING):
    """Exactly the notebook's encoder (cell 23), refit inside each fold."""
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
    n0 = n_in - len(df)
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
    for c in EXCLUDED_FROM_MODEL:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X, y, df["Address"], (n_in, n0, n1 + n2, len(df))


def run(X, y, groups, hazard_keep, seed):
    """GroupKFold spatial CV; target encoding refit inside every fold."""
    cols = [c for c in X.columns if c in hazard_keep or c not in HAZARD_COLS]
    Xc = X[cols]
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = []
    for tr, va in gkf.split(Xc, y, groups=groups):
        Xtr, Xva = Xc.iloc[tr].copy(), Xc.iloc[va].copy()
        ytr, yva = y.iloc[tr], y.iloc[va]
        gtr, gva = groups.iloc[tr], groups.iloc[va]
        oof, enc = target_encode_oof(gtr, ytr, gva, seed)
        Xtr["Address_target_enc"] = oof.values
        Xva["Address_target_enc"] = enc.values
        Xtr.columns = [clean_name(c) for c in Xtr.columns]
        Xva.columns = [clean_name(c) for c in Xva.columns]
        m = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed,
                                  n_jobs=-1)
        m.fit(Xtr, ytr)
        scores.append(r2_score(yva, m.predict(Xva)) * 100)
    return np.array(scores)


def main():
    started = dt.datetime.now()
    X, y, groups, (n_in, n_floor, n_dropped, n_rows) = build_matrix()
    configs = {"baseline_all5_hazard": set(HAZARD_COLS), "no_hazard": set()}
    for h in HAZARD_COLS:
        configs[f"drop_{h}"] = set(HAZARD_COLS) - {h}

    print(f"rows in {n_in} -> price floor/ceiling removed {n_floor} -> "
          f"trimmed {n_dropped} at k={K_TRIM} -> {n_rows}")
    print(f"features (before target encoding): {X.shape[1]}")
    print(f"groups (distinct Address): {groups.nunique()}")
    print(f"seeds: {SEEDS}\n")

    rows = []
    for cfg, keep in configs.items():
        for seed in SEEDS:
            s = run(X, y, groups, keep, seed)
            rows.append({
                "config": cfg, "seed": seed,
                **{f"fold{i+1}_r2": round(v, 4) for i, v in enumerate(s)},
                "mean_r2": round(s.mean(), 4), "sd_across_folds": round(s.std(ddof=1), 4),
            })
            print(f"  {cfg:28s} seed {seed:>4}  mean {s.mean():6.2f}  "
                  f"folds {np.round(s, 2)}")

    runs = pd.DataFrame(rows)
    fold_cols = [f"fold{i+1}_r2" for i in range(N_SPLITS)]

    summ = []
    base = runs[runs.config == "baseline_all5_hazard"]
    base_by_seed = base.set_index("seed")["mean_r2"]
    for cfg in configs:
        sub = runs[runs.config == cfg]
        by_seed = sub.set_index("seed")["mean_r2"]
        delta = (by_seed - base_by_seed) if cfg != "baseline_all5_hazard" else by_seed * 0
        # sign consistency: how many seeds move the same way as the mean
        same = int((np.sign(delta) == np.sign(delta.mean())).sum()) if delta.mean() else 0
        summ.append({
            "config": cfg,
            "mean_r2": round(by_seed.mean(), 4),
            "sd_across_seeds": round(by_seed.std(ddof=1), 4),
            "min_seed_mean": round(by_seed.min(), 4),
            "max_seed_mean": round(by_seed.max(), 4),
            "delta_vs_baseline_pp": round(delta.mean(), 4),
            "delta_sd_pp": round(delta.std(ddof=1), 4),
            "delta_seeds_same_sign": same,
            "above_noise_floor": bool(abs(delta.mean()) > 2 * delta.std(ddof=1))
                                 if cfg != "baseline_all5_hazard" else "",
            **{f"{c}_mean": round(sub[c].mean(), 4) for c in fold_cols},
        })
    summary = pd.DataFrame(summ)

    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by src/step7_hazard_ablation.py\n"
            f"# dataset={os.path.basename(DATA)} rows_in={n_in} "
            f"price_floor_ceiling_removed={n_floor} trimmed_k={K_TRIM} "
            f"rows_used={n_rows} groups={groups.nunique()}\n"
            f"# model=RandomForestRegressor(n_estimators={N_ESTIMATORS}, n_jobs=-1) "
            f"cv=GroupKFold(n_splits={N_SPLITS}) by Address\n"
            f"# target=log(Price per Perch) encoder=oof_target_enc(KFold5, "
            f"smoothing={SMOOTHING}) refit per fold\n"
            f"# hazard_set={'|'.join(HAZARD_COLS)} excluded={'|'.join(EXCLUDED_FROM_MODEL)}\n"
            f"# seeds={'|'.join(map(str, SEEDS))}\n"
            f"# NOTE GroupKFold is deterministic - folds identical across seeds; "
            f"seed varies RF and the target encoder only\n")
    for path, frame in ((OUT_RUNS, runs), (OUT_SUM, summary)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(meta)
            frame.to_csv(fh, index=False)

    write_report(runs, summary, started, n_in, n_floor, n_dropped, n_rows, groups, X)
    print(f"\nwrote {OUT_RUNS}\nwrote {OUT_SUM}\nwrote {OUT_MD}")


def write_report(runs, summary, started, n_in, n_floor, n_dropped, n_rows, groups, X):
    s = summary.set_index("config")
    base_mean = s.loc["baseline_all5_hazard", "mean_r2"]
    base_sd = s.loc["baseline_all5_hazard", "sd_across_seeds"]
    noise = 2 * base_sd

    md = [f"# Hazard ablation - {len(runs.seed.unique())} seeds, spatial CV", "",
          f"Generated {started:%Y-%m-%d %H:%M} by `src/step7_hazard_ablation.py`. "
          "This replaces `model test/hazard_ablation_results.csv`, which had no "
          "script behind it.", "",
          "## Specification", "",
          "| | |", "|---|---|",
          f"| dataset | `final_dataset_hazard.csv` |",
          f"| rows in / price floor-ceiling removed / trimmed at k={K_TRIM} / used | "
          f"{n_in} / {n_floor} / {n_dropped} / {n_rows} |",
          f"| groups (distinct Address) | {groups.nunique()} |",
          f"| features before target encoding | {X.shape[1]} |",
          f"| model | `RandomForestRegressor(n_estimators={N_ESTIMATORS}, n_jobs=-1)` |",
          f"| CV | `GroupKFold(n_splits={N_SPLITS})` grouped by `Address` |",
          f"| target encoding | out-of-fold, KFold(5), smoothing={SMOOTHING}, **refit inside each spatial fold** |",
          f"| hazard set | {', '.join('`' + h + '`' for h in HAZARD_COLS)} |",
          f"| excluded | {', '.join('`' + h + '`' for h in EXCLUDED_FROM_MODEL)} (in the file, not in the modelling set) |",
          f"| seeds | {', '.join(map(str, sorted(runs.seed.unique())))} |", "",
          "> **What the seed varies.** `GroupKFold` has no `random_state`; it "
          "partitions groups deterministically, so **the five folds are identical "
          "across all seeds**. A seed changes only the RandomForest and the inner "
          "KFold of the target encoder. The spread below is therefore model "
          "stochasticity and is a **lower bound** on true run-to-run variance - "
          "fold-assignment variance is not captured.", "",
          "## Results", "",
          "| config | mean R² | SD across seeds | range | Δ vs baseline (pp) | Δ SD (pp) |",
          "|---|---|---|---|---|---|"]
    for cfg in s.index:
        r = s.loc[cfg]
        d = "—" if cfg == "baseline_all5_hazard" else f"{r['delta_vs_baseline_pp']:+.2f}"
        dsd = "—" if cfg == "baseline_all5_hazard" else f"{r['delta_sd_pp']:.2f}"
        md.append(f"| `{cfg}` | **{r['mean_r2']:.2f}** | {r['sd_across_seeds']:.2f} | "
                  f"{r['min_seed_mean']:.2f}–{r['max_seed_mean']:.2f} | {d} | {dsd} |")

    md += ["", "## Per-fold means (averaged over seeds)", "",
           "| config | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |",
           "|---|---|---|---|---|---|"]
    for cfg in s.index:
        r = s.loc[cfg]
        md.append(f"| `{cfg}` | " + " | ".join(
            f"{r[f'fold{i+1}_r2_mean']:.2f}" for i in range(N_SPLITS)) + " |")

    md += ["", "## Noise floor - which deltas are real", "",
           f"The baseline alone varies by **{base_sd:.2f} pp SD** across seeds. But "
           "the correct test for a leave-one-out ablation is the **paired** delta: "
           "each config is compared against the baseline *on the same seed*, which "
           "cancels the seed-level shift they share. A delta counts as real only if "
           "it exceeds **2 SD of its own paired distribution**, and the sign should "
           "be consistent across seeds.", "",
           "| config | Δ (pp) | SD of Δ | 2 SD | seeds agreeing on sign | above floor? |",
           "|---|---|---|---|---|---|"]
    for cfg in s.index:
        if cfg == "baseline_all5_hazard":
            continue
        d = s.loc[cfg, "delta_vs_baseline_pp"]
        sd = s.loc[cfg, "delta_sd_pp"]
        ok = abs(d) > 2 * sd
        md.append(f"| `{cfg}` | {d:+.2f} | {sd:.2f} | {2 * sd:.2f} | "
                  f"{int(s.loc[cfg, 'delta_seeds_same_sign'])}/{len(runs.seed.unique())} | "
                  f"{'**yes**' if ok else 'no'} |")
    md += ["", "Because `GroupKFold` fixes the folds, even these paired SDs "
           "understate the true variance - a different fold partition would move "
           "the numbers further.", ""]

    # the three prior claims
    b1 = s.loc["baseline_all5_hazard", "fold1_r2_mean"]
    n1 = s.loc["no_hazard", "fold1_r2_mean"]
    b2 = s.loc["baseline_all5_hazard", "fold2_r2_mean"]
    n2 = s.loc["no_hazard", "fold2_r2_mean"]
    gain = s.loc["baseline_all5_hazard", "mean_r2"] - s.loc["no_hazard", "mean_r2"]
    woc = s.loc["drop_water_occurrence_pct", "delta_vs_baseline_pp"]

    md += ["", "## Do the single-seed claims survive?", "",
           "| claim | single-seed value | multi-seed value | survives? |",
           "|---|---|---|---|",
           f"| hazard block gain ({PRIOR['no_hazard']:.2f} → {PRIOR['baseline_all5_hazard']:.2f}) | "
           f"+{PRIOR['baseline_all5_hazard'] - PRIOR['no_hazard']:.2f} pp | "
           f"{gain:+.2f} pp (SD {s.loc['no_hazard', 'delta_sd_pp']:.2f}, "
           f"{int(s.loc['no_hazard', 'delta_seeds_same_sign'])}/5 seeds agree) | "
           f"{'**yes, but marginally**' if s.loc['no_hazard', 'above_noise_floor'] else 'NO'} |",
           f"| `water_occurrence_pct` costs 2.12 pp | −2.12 pp | {woc:+.2f} pp (SD "
           f"{s.loc['drop_water_occurrence_pct', 'delta_sd_pp']:.2f}, "
           f"{int(s.loc['drop_water_occurrence_pct', 'delta_seeds_same_sign'])}/5 agree) | "
           f"{'**yes, but at the threshold**' if s.loc['drop_water_occurrence_pct', 'above_noise_floor'] else 'NO'} |",
           f"| fold 1 regression | −5.06 pp | {b1 - n1:+.2f} pp | "
           f"{'**yes**' if (b1 - n1) < 0 else 'NO - sign flips'} |",
           f"| fold 2 gain | +12.49 pp | {b2 - n2:+.2f} pp | "
           f"{'**yes**' if (b2 - n2) > 0 else 'NO - sign flips'} |", "",
           "## Comparison with the unreproducible file", "",
           "| config | `hazard_ablation_results.csv` (1 seed, no script) | this run (multi-seed) | difference |",
           "|---|---|---|---|"]
    for cfg in s.index:
        if cfg in PRIOR:
            md.append(f"| `{cfg}` | {PRIOR[cfg]:.2f} | {s.loc[cfg, 'mean_r2']:.2f} | "
                      f"{s.loc[cfg, 'mean_r2'] - PRIOR[cfg]:+.2f} |")
    md += ["", "Full per-seed values: `hazard_ablation_multiseed.csv`. "
           "Per-config summary: `hazard_ablation_summary.csv`. Both carry their "
           "specification in the header lines.", ""]

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())

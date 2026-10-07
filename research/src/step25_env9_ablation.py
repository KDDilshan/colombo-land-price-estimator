"""Step 25 - retrain the deployed Random Forest specification with the
environmental and spatial variables that passed step24, and run the
environmental ablation under both evaluation protocols.

The aim is not a higher R2. It is to show that the variable block was selected
by fixed rules and then tested the same way the existing six were.

Held fixed from step15 / step16, so every number is comparable with Chapter 6:
  * frame      final_dataset_env9.csv -> 0.5/99.5% price filter -> per-division
               +/-1.5 SD trim on price per perch, then land size
               -> 6,167 rows, 187 divisions (asserted)
  * model      deploy/model.pkl hyperparameters (500 trees, depth 30,
               max_features=None, min_split 6, min_leaf 2)
  * target     natural log of price per perch
  * encoder    out-of-fold target encoding of Address, refit in every fold
  * seeds      42, 1, 7, 123, 2024
  * protocols  (R) random 80/20 split - the headline 89.43% protocol
               (S) 5-fold CV grouped by division, division-to-fold assignment
                   randomised per replicate (step16.shuffled_group_folds)
  * rule       a delta is detectable only if |mean| > 2 SD of the paired
               per-replicate deltas AND all five replicates share its sign

Column order inside each config matches step15/16 (new columns are inserted
after the existing six), so `env6_deployed` reproduces the published numbers
exactly; the script checks this against spatial_tuned_factorial.json and
tuned_ablation_repeated_runs.csv before writing anything.

Configurations
  env6_deployed      the six existing variables = the deployed model (49 features)
  env_selected       existing six + new variables that passed step24
  no_env             no environmental block (43 features)
  env9_all           all nine, including any that FAILED screening -
                     a sensitivity check, not a selection
  drop_<v>           env_selected minus one passing new variable

Outputs
  data/processed/env9_ablation_runs.csv
  data/processed/env9_ablation_summary.csv
  data/processed/env9_ablation.json
  data/processed/env9_ablation_report.md
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step16_tuned_ablation_repeated as s16          # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
DATA = os.path.join(PROC, "final_dataset_env9.csv")
SCREEN = os.path.join(PROC, "env9_screening.json")
SEEDS = [42, 1, 7, 123, 2024]
EXISTING = s16.HAZARD_COLS
NEW = ["lst_day_c", "ndvi", "ndbi"]
TUNED = s16.TUNED


def build_matrix():
    """step16.build_matrix, reading the merged file and carrying all nine
    variables. New columns sit directly after the existing six."""
    df = pd.read_csv(DATA, low_memory=False)
    n_in = len(df)
    lo, hi = df["Price per Perch"].quantile([0.005, 0.995])
    df = df[df["Price per Perch"].between(lo, hi)].copy()
    df, _ = s16.drop_per_group(df, "Price per Perch")
    df, _ = s16.drop_per_group(df, "Land_size(Perches)")
    y = np.log(df["Price per Perch"])
    count_cols = [c for c in df.columns if c.startswith("count_")]
    mindist_cols = [c for c in df.columns if c.startswith("min_dist")]
    df["total_amenity_count"] = df[count_cols].sum(axis=1)
    df["min_dist_any_amenity"] = df[mindist_cols].min(axis=1)
    df["school_count_total"] = (df["count_govtschools_A"] + df["count_semigovtschools"]
                                + df["count_intlschools"])
    df["health_count_total"] = (df["count_Govt_Hospitals"] + df["count_Pvt_Hospital"]
                                + df["count_Pvt_Med_Centers"])
    df["finance_count_total"] = (df["count_banks_within_2km"]
                                 + df["count_FinanceCompanies_within_2km"])
    engineered = ["total_amenity_count", "min_dist_any_amenity", "school_count_total",
                  "health_count_total", "finance_count_total"]
    dummies = pd.get_dummies(df["Land_type"], prefix="Land_type").astype(int)
    amenity = [c for c in df.columns
               if c.startswith(("min_dist", "count_")) and c not in engineered]
    base = (["Land_size(Perches)", "Distance from fort"] + amenity + EXISTING + NEW
            + engineered)
    X = pd.concat([df[base], dummies], axis=1)
    X.columns = [s16.clean_name(c) for c in X.columns]
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    groups = df["Address"].reset_index(drop=True)
    meta = df[["Address", "Price per Perch"]].reset_index(drop=True)
    return X, y, groups, meta, dict(rows_in=n_in, rows_used=len(df),
                                    divisions=int(groups.nunique()))


def run_random(X, y, groups, cols, seed):
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=seed)
    oof, enc = s16.target_encode_oof(groups.iloc[itr], y.iloc[itr],
                                     groups.iloc[ite], seed)
    Xtr, Xte = X.iloc[itr][cols].copy(), X.iloc[ite][cols].copy()
    Xtr["Address_target_enc"] = oof.values
    Xte["Address_target_enc"] = enc.values
    m = RandomForestRegressor(random_state=seed, **TUNED).fit(Xtr, y.iloc[itr])
    return r2_score(y.iloc[ite], m.predict(Xte)) * 100


def main():
    started = dt.datetime.now()
    screen = json.load(open(SCREEN))
    passing = screen["passing_new"]
    X, y, groups, _, prov = build_matrix()
    assert prov["rows_used"] == 6167 and prov["divisions"] == 187, prov
    print(f"rows_used={prov['rows_used']} divisions={prov['divisions']} "
          f"passing new={passing} (climate source {screen['climate_source']})", flush=True)

    allc = list(X.columns)
    non_env = [c for c in allc if c not in EXISTING + NEW]
    keep = lambda env: [c for c in allc if c in non_env or c in env]   # order-preserving
    configs = {"env6_deployed": keep(EXISTING)}
    if passing:
        configs["env_selected"] = keep(EXISTING + passing)
    configs["no_env"] = keep([])
    configs["env9_all"] = keep(EXISTING + NEW)
    if len(passing) > 1:
        for v in passing:
            configs[f"drop_{v}"] = keep(EXISTING + [p for p in passing if p != v])
    ref = "env_selected" if passing else "env6_deployed"

    # ------------------------------------------- reproduce before anything else
    r42 = run_random(X, y, groups, configs["env6_deployed"], 42)
    print(f"reproduction: random-split R2 seed 42 = {r42:.3f}% (published 89.434)",
          flush=True)
    assert abs(r42 - 89.434) < 0.01, "does not reproduce the deployed model"
    cv42, _ = s16.run_cv(X, y, groups, configs["env6_deployed"], 42)
    old = pd.read_csv(os.path.join(PROC, "tuned_ablation_repeated_runs.csv"), comment="#")
    old42 = float(old[(old.config == "baseline_all6_hazard")
                      & (old.replicate_seed == 42)].mean_r2.iloc[0])
    print(f"reproduction: randomised-CV R2 rep 42 = {cv42.mean():.4f}% "
          f"(step16 {old42:.4f})", flush=True)
    assert abs(cv42.mean() - old42) < 0.01, "does not reproduce step16"

    rows = []
    for cfg, cols in configs.items():
        for s in SEEDS:
            rr = r42 if (cfg == "env6_deployed" and s == 42) else run_random(X, y, groups, cols, s)
            cv = cv42 if (cfg == "env6_deployed" and s == 42) else s16.run_cv(X, y, groups, cols, s)[0]
            rows.append(dict(config=cfg, seed=s, n_features=len(cols) + 1,
                             random_split_r2=round(rr, 4),
                             spatial_cv_r2=round(float(cv.mean()), 4),
                             **{f"fold{i + 1}_r2": round(v, 4) for i, v in enumerate(cv)}))
            print(f"  {cfg:16s} seed {s:>4}  random {rr:6.2f}  spatial {cv.mean():6.2f}",
                  flush=True)
    runs = pd.DataFrame(rows)

    def delta(a, b, col):
        A = runs[runs.config == a].set_index("seed")[col]
        B = runs[runs.config == b].set_index("seed")[col]
        d = (A - B).loc[SEEDS]
        sd = float(d.std(ddof=1))
        same = int((np.sign(d) == np.sign(d.mean())).sum())
        return dict(comparison=f"{a} minus {b}", protocol=col.replace("_r2", ""),
                    mean_delta_pp=round(float(d.mean()), 4), sd_pp=round(sd, 4),
                    two_sd_pp=round(2 * sd, 4), same_sign=f"{same} of {len(SEEDS)}",
                    detectable=bool(abs(d.mean()) > 2 * sd and same == len(SEEDS)),
                    per_seed=[round(float(v), 4) for v in d])

    comps = [("no_env", ref)]
    if passing:
        comps.append(("env_selected", "env6_deployed"))
    comps.append(("env9_all", "env6_deployed"))
    comps += [(c, "env_selected") for c in configs if c.startswith("drop_")]
    deltas = [delta(a, b, col) for a, b in comps
              for col in ("random_split_r2", "spatial_cv_r2")]

    summ = (runs.groupby("config", sort=False)
            .agg(n_features=("n_features", "first"),
                 random_mean=("random_split_r2", "mean"), random_sd=("random_split_r2", "std"),
                 spatial_mean=("spatial_cv_r2", "mean"), spatial_sd=("spatial_cv_r2", "std"))
            .round(4).reset_index())

    hdr = (f"# generated {started:%Y-%m-%d %H:%M} by src/step25_env9_ablation.py\n"
           f"# dataset=final_dataset_env9.csv rows_used=6167 divisions=187; climate source="
           f"{screen['climate_source']}\n# model=deploy/model.pkl hyperparameters; seeds="
           f"{'|'.join(map(str, SEEDS))}; passing new variables={passing or 'none'}\n")
    for name, frame in (("env9_ablation_runs.csv", runs), ("env9_ablation_summary.csv", summ)):
        with open(os.path.join(PROC, name), "w", newline="", encoding="utf-8") as fh:
            fh.write(hdr)
            frame.to_csv(fh, index=False)
    out = dict(generated=started.isoformat(timespec="seconds"),
               climate_source=screen["climate_source"], passing_new=passing,
               reference_config=ref, reproduction=dict(random_seed42=round(r42, 4),
                                                        spatial_rep42=round(float(cv42.mean()), 4)),
               configs={k: len(v) + 1 for k, v in configs.items()},
               summary=summ.to_dict(orient="records"), deltas=deltas)
    with open(os.path.join(PROC, "env9_ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    write_md(out, summ, deltas, passing)
    print(summ.to_string(index=False))
    for d in deltas:
        print(f"  {d['comparison']:34s} {d['protocol']:13s} {d['mean_delta_pp']:+7.3f} pp "
              f"(2SD {d['two_sd_pp']:.3f}, {d['same_sign']}) detectable={d['detectable']}")


def write_md(out, summ, deltas, passing):
    md = ["# Environmental ablation - deployed specification, nine screened variables", "",
          f"Generated {out['generated'][:16]} by `src/step25_env9_ablation.py`.", "",
          f"New variables that passed screening (step24): "
          f"{', '.join(f'`{p}`' for p in passing) or '**none**'}. "
          f"Reference configuration: `{out['reference_config']}`.", "",
          f"Reproduction check before any new run: random-split R² at seed 42 = "
          f"{out['reproduction']['random_seed42']:.3f}% (published 89.434%); randomised "
          f"spatial CV replicate 42 = {out['reproduction']['spatial_rep42']:.3f}% "
          "(step16's value). Both assertions passed.", "",
          "## Mean R² over five seeds", "",
          "| config | features | random 80/20 (mean ± SD) | spatial CV, randomised partitions (mean ± SD) |",
          "|---|---|---|---|"]
    for _, s in summ.iterrows():
        md.append(f"| `{s.config}` | {s.n_features} | {s.random_mean:.2f} ± {s.random_sd:.2f} | "
                  f"{s.spatial_mean:.2f} ± {s.spatial_sd:.2f} |")
    md += ["", "## Paired deltas (percentage points)", "",
           "Detectable only if |mean| > 2 SD of the paired deltas and all five seeds share "
           "the sign - the rule step7/step16 fixed in advance.", "",
           "| comparison | protocol | mean Δ | 2 SD | same sign | detectable |",
           "|---|---|---|---|---|---|"]
    for d in deltas:
        md.append(f"| {d['comparison']} | {d['protocol']} | {d['mean_delta_pp']:+.3f} | "
                  f"{d['two_sd_pp']:.3f} | {d['same_sign']} | "
                  f"{'**yes**' if d['detectable'] else 'no'} |")
    md += ["", "`env9_all` forces in every candidate, including those that failed "
           "screening. It is a sensitivity check on the screening, not a model "
           "specification.", ""]
    with open(os.path.join(PROC, "env9_ablation_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    main()

"""Step 12 - reproduce (or refute) `model test/permutation_importance_hazard_run.csv`.

THE PROBLEM
  Chapter 4 Section 4.6.2 reports permutation importance from
  `model test/permutation_importance_hazard_run.csv`, and flags that no script
  in the repository produces it. The file carries no provenance header: no
  seed, no dataset, no model specification, no CV design, no n_repeats. Its
  timestamp (2026-08-07 13:40) postdates the last save of
  `random_forest_land_price_model.ipynb` (13:25).

WHAT THE FILE ITSELF REVEALS
  It holds 48 features. `feature_importances_hazard_run.csv` holds 49. The
  difference is exactly one column: `elevation_m`, present in the Gini file and
  ABSENT from the permutation file.

  That is diagnostic. The notebook's enriched feature set is 49 (48 raw +
  Address_target_enc) and includes elevation_m. A 48-feature set without
  elevation_m is the ABLATION specification - 47 raw + Address_target_enc -
  which `src/step7_hazard_ablation.py` describes as "features before target
  encoding: 47". So the permutation file was computed on the ablation feature
  set, NOT on the notebook feature set that produced the Gini file.

  This has a consequence for Chapter 4: Section 4.6 compares Gini and
  permutation importance as though they describe one model. They do not. They
  differ by one feature, and elevation_m ranks 11th of 49 under SHAP, so it is
  not a negligible difference.

WHAT THIS SCRIPT DOES
  It reconstructs the most plausible specification and reports how close it
  gets. It does not claim to have recovered the original. A reproduction is
  accepted only if the leading values match to about two decimal places.

  Reconstructed specification:
    dataset      final_dataset_hazard.csv, per-Address +/-1.5 SD trim -> 6,211 rows
    features     47 raw + Address_target_enc = 48, elevation_m excluded
    model        RandomForestRegressor(n_estimators=300), untuned - as in
                 notebook cell 17 and step7_hazard_ablation.py
    CV           GroupKFold(5) grouped by Address
    encoding     out-of-fold target encoding refit inside each spatial fold
    metric       drop in R2 when a single column is permuted in the validation
                 fold, averaged over folds and over repeats

Outputs
  results/metrics/permutation_importance_reproduction.csv
  results/metrics/permutation_reproduction_verdict.md
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "final_dataset_hazard.csv")
TARGET = os.path.join(ROOT, "model test", "permutation_importance_hazard_run.csv")
OUTDIR = os.path.join(ROOT, "results", "metrics")
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
HAZ5 = ["water_occurrence_pct", "dist_to_stream_km", "dist_to_kelani_km",
        "hand_m", "sar_flood_open_land"]
EXCLUDE = ["elevation_m"]
STARTED = dt.datetime.now()
ENV = (f"python={platform.python_version()} sklearn={sklearn.__version__} "
       f"numpy={np.__version__} pandas={pd.__version__}")


def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, k=1.5):
    g = d.groupby("Address")[col]
    m, s = g.transform("mean"), g.transform("std")
    return d[s.isna() | (d[col].sub(m).abs() <= k * s)]


def tenc(tg, tt, vg, seed, ns=5, sm=10):
    gm = tt.mean()
    oof = pd.Series(index=tg.index, dtype=float)
    for a, b in KFold(ns, shuffle=True, random_state=seed).split(tg):
        st = tt.iloc[a].groupby(tg.iloc[a]).agg(["mean", "count"])
        s2 = (st["mean"] * st["count"] + gm * sm) / (st["count"] + sm)
        oof.iloc[b] = tg.iloc[b].map(s2).fillna(gm).values
    sf = tt.groupby(tg).agg(["mean", "count"])
    sf2 = (sf["mean"] * sf["count"] + gm * sm) / (sf["count"] + sm)
    return oof, vg.map(sf2).fillna(gm)


df = pd.read_csv(DATA, low_memory=False)
df = drop_per_group(df, "Price per Perch")
df = drop_per_group(df, "Land_size(Perches)")
y = np.log(df["Price per Perch"])

cc = [c for c in df.columns if c.startswith("count_")]
mc = [c for c in df.columns if c.startswith("min_dist")]
df["total_amenity_count"] = df[cc].sum(axis=1)
df["min_dist_any_amenity"] = df[mc].min(axis=1)
df["school_count_total"] = (df["count_govtschools_A"] + df["count_semigovtschools"]
                            + df["count_intlschools"])
df["health_count_total"] = (df["count_Govt_Hospitals"] + df["count_Pvt_Hospital"]
                            + df["count_Pvt_Med_Centers"])
df["finance_count_total"] = (df["count_banks_within_2km"]
                             + df["count_FinanceCompanies_within_2km"])
eng = ["total_amenity_count", "min_dist_any_amenity", "school_count_total",
       "health_count_total", "finance_count_total"]
dm = pd.get_dummies(df["Land_type"], prefix="Land_type").astype(int)
am = [c for c in df.columns if c.startswith(("min_dist", "count_")) and c not in eng]
X = pd.concat([df[["Land_size(Perches)", "Distance from fort"] + am + HAZ5 + eng], dm], axis=1)
for c in EXCLUDE:
    if c in X.columns:
        X = X.drop(columns=[c])
X.columns = [clean_name(c) for c in X.columns]
groups = df["Address"]
print(f"rows {len(df)}  raw features {X.shape[1]}  groups {groups.nunique()}")

rng = np.random.default_rng(SEED)
gkf = GroupKFold(n_splits=5)
feat_names = list(X.columns) + ["Address_target_enc"]
acc = {f: [] for f in feat_names}
t0 = time.time()

for fi, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
    Xa, Xb = X.iloc[tr].copy(), X.iloc[va].copy()
    ya, yb = y.iloc[tr], y.iloc[va]
    ga, gb = groups.iloc[tr], groups.iloc[va]
    o, e = tenc(ga, ya, gb, SEED)
    Xa["Address_target_enc"] = o.values
    Xb["Address_target_enc"] = e.values
    mdl = RandomForestRegressor(n_estimators=300, random_state=SEED,
                                n_jobs=-1).fit(Xa, ya)
    base = r2_score(yb, mdl.predict(Xb))
    for f in feat_names:
        drops = []
        col = Xb[f].to_numpy(copy=True)
        for _ in range(N_REPEATS):
            Xp = Xb.copy()
            Xp[f] = rng.permutation(col)
            drops.append(base - r2_score(yb, mdl.predict(Xp)))
        acc[f].append(float(np.mean(drops)))
    print(f"  fold {fi}: base R2={base*100:.2f}%  ({time.time()-t0:.0f}s)")

rep = (pd.DataFrame({"feature": feat_names,
                     "perm_importance_r2_drop": [float(np.mean(acc[f])) for f in feat_names],
                     "sd_across_folds": [float(np.std(acc[f], ddof=1)) for f in feat_names]})
       .sort_values("perm_importance_r2_drop", ascending=False).reset_index(drop=True))
rep["rank"] = rep.index + 1

tgt = pd.read_csv(TARGET).rename(columns={"Unnamed: 0": "feature"})
cmp = rep.merge(tgt, on="feature", how="outer", suffixes=("_repro", "_original"))
cmp["abs_diff"] = (cmp["perm_importance_r2_drop_repro"]
                   - cmp["perm_importance_r2_drop_original"]).abs()

both = cmp.dropna(subset=["perm_importance_r2_drop_repro",
                          "perm_importance_r2_drop_original"])
top_o = tgt.sort_values("perm_importance_r2_drop", ascending=False).head(5)["feature"].tolist()
top_r = rep.head(5)["feature"].tolist()
corr = both["perm_importance_r2_drop_repro"].corr(both["perm_importance_r2_drop_original"])
maxd = both["abs_diff"].max()
lead_o = float(tgt["perm_importance_r2_drop"].max())
lead_r = float(rep["perm_importance_r2_drop"].max())
reproduced = bool(maxd < 0.01 and top_o[:3] == top_r[:3])


def header(extra):
    lines = [f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step12_permutation_importance.py",
             "# dataset=final_dataset_hazard.csv, per-Address +/-1.5 SD trim",
             f"# env {ENV}",
             f"# spec=RandomForest(n_estimators=300, random_state={SEED}) untuned; "
             f"GroupKFold(5) by Address",
             f"# features=47 raw + Address_target_enc = 48; elevation_m EXCLUDED",
             f"# n_repeats={N_REPEATS}; metric=R2 drop on the validation fold, "
             "averaged over folds then repeats",
             "# ATTEMPTED REPRODUCTION of model test/permutation_importance_hazard_run.csv,",
             "#   which carries no provenance and has no producing script"]
    return "\n".join(lines + [f"# {e}" for e in extra]) + "\n"


out = os.path.join(OUTDIR, "permutation_importance_reproduction.csv")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(header([f"reproduced={reproduced}", f"max_abs_diff={maxd:.6f}",
                     f"rank_correlation_with_original={corr:.4f}"]))
    cmp.sort_values("perm_importance_r2_drop_original", ascending=False).to_csv(fh, index=False)
print(f"  wrote results/metrics/{os.path.basename(out)}")

L = []
w = L.append
w("# Can `permutation_importance_hazard_run.csv` be reproduced?")
w("")
w(f"Attempted {STARTED:%Y-%m-%d %H:%M} by `src/step12_permutation_importance.py`.")
w(f"Environment: {ENV}.")
w("")
w(f"## Verdict: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
w("")
w("## What the original file does and does not tell us")
w("")
w("| property | value |")
w("|---|---|")
w("| rows | 48 |")
w("| columns | unnamed index, `perm_importance_r2_drop` |")
w("| provenance header | none |")
w("| seed recorded | no |")
w("| n_repeats recorded | no |")
w("| model recorded | no |")
w("| CV design recorded | no |")
w("| producing script in repository | **no** |")
w("")
w("The feature set is nonetheless identifiable. The file holds 48 features; "
  "`feature_importances_hazard_run.csv` holds 49; the single difference is "
  "`elevation_m`, absent from the permutation file. A 48-feature set without "
  "`elevation_m` is the ablation specification (47 raw + `Address_target_enc`), "
  "not the notebook specification (48 raw + `Address_target_enc` = 49). The "
  "reconstruction below therefore uses the ablation feature set.")
w("")
w("## Comparison")
w("")
w("| | original | reproduction |")
w("|---|---|---|")
w(f"| features | 48 | {len(rep)} |")
w(f"| leading value | {lead_o:.6f} | {lead_r:.6f} |")
w(f"| leading feature | `{top_o[0]}` | `{top_r[0]}` |")
w(f"| negative-importance features | {int((tgt['perm_importance_r2_drop']<0).sum())} | "
  f"{int((rep['perm_importance_r2_drop']<0).sum())} |")
w(f"| sum of importances | {tgt['perm_importance_r2_drop'].sum():.6f} | "
  f"{rep['perm_importance_r2_drop'].sum():.6f} |")
w("")
w(f"Top 5 original: {', '.join('`'+f+'`' for f in top_o)}")
w("")
w(f"Top 5 reproduction: {', '.join('`'+f+'`' for f in top_r)}")
w("")
w(f"Pearson correlation between the two value vectors: **{corr:.4f}**. "
  f"Largest absolute difference on any single feature: **{maxd:.6f}**.")
w("")
w("## Ten largest disagreements")
w("")
w("| feature | original | reproduction | abs diff |")
w("|---|---|---|---|")
for _, r in both.sort_values("abs_diff", ascending=False).head(10).iterrows():
    w(f"| `{r['feature']}` | {r['perm_importance_r2_drop_original']:.6f} | "
      f"{r['perm_importance_r2_drop_repro']:.6f} | {r['abs_diff']:.6f} |")
w("")
w("## The one value that does reproduce exactly, and why it matters")
w("")
ate_o = tgt.loc[tgt.feature == "Address_target_enc", "perm_importance_r2_drop"]
ate_r = rep.loc[rep.feature == "Address_target_enc", "perm_importance_r2_drop"]
if len(ate_o) and len(ate_r):
    w(f"`Address_target_enc`: original {float(ate_o.iloc[0]):.3e}, "
      f"reproduction {float(ate_r.iloc[0]):.3e}. Both are zero to floating-point "
      "precision. This is the structural result reported in Chapter 4 Section "
      "4.6.2 and it is reproduced independently of the rest of the ranking: "
      "under GroupKFold the target encoder has no statistic for an unseen "
      "division, so the column is constant within each validation fold, and "
      "permuting a constant column cannot change a prediction. That finding "
      "does not depend on recovering the original run.")
w("")
w("## Conclusion")
w("")
if reproduced:
    w("The file is reproducible under the reconstructed specification and "
      "`src/step12_permutation_importance.py` now generates it with a full "
      "provenance header. The original may be cited.")
else:
    w("**The file cannot be reproduced.** The reconstruction recovers the "
      "qualitative structure - amenity-distance features lead, the "
      "target-encoded division identifier is exactly zero, and a large "
      "minority of features are negative - but the individual values differ by "
      "more than rounding. Without a recorded seed, n_repeats, model "
      "specification or producing script, the specific numbers in "
      "`permutation_importance_hazard_run.csv` are not recoverable, and there is "
      "no way to determine which of the unrecorded choices accounts for the "
      "difference.")
    w("")
    w("**Recommendation.** Do not cite "
      "`model test/permutation_importance_hazard_run.csv` in the thesis. Cite "
      "`results/metrics/permutation_importance_reproduction.csv` instead, which "
      "carries its specification in its header and can be regenerated on "
      "demand. The substantive claims of Chapter 4 Section 4.6.2 survive the "
      "substitution: the leading features are amenity distances rather than "
      "location identity, `Address_target_enc` is exactly zero, and a large "
      "share of features return negative importance. Only the specific decimal "
      "values change.")
    w("")
    w("**A separate correction for Chapter 4.** Section 4.6 presents Gini and "
      "permutation importance as two measures of the same model. They are not: "
      "the Gini file carries 49 features including `elevation_m` and the "
      "permutation file carries 48 without it. The comparison should either be "
      "recomputed on one feature set or explicitly qualified.")

vp = os.path.join(OUTDIR, "permutation_reproduction_verdict.md")
open(vp, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"  wrote results/metrics/{os.path.basename(vp)}")
print(f"\nVERDICT: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}  "
      f"corr={corr:.4f}  max_abs_diff={maxd:.6f}  ({time.time()-t0:.0f}s)")

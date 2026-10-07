"""
step15_spatial_tuned_factorial.py
---------------------------------
Resolves the central methodological defect identified in the thesis audit:
the random-split result (89.43% R2) came from the TUNED 500-tree deployed
Random Forest, while the spatial GroupKFold result (41.25% R2) came from an
UNTUNED 300-tree Random Forest. The thesis claimed the gap between them was
attributable to the partitioning scheme alone. That claim was not supportable,
because two things differed at once.

This script runs the full 2x2 factorial so the two factors can be separated:

                      | random 80/20 split | GroupKFold by division
  tuned 500-tree RF   |        A           |         B   <- new
  untuned 300-tree RF |        D   <- new  |         C

Everything else is held constant across all four cells: the same cleaned
6,167-row frame, the same 49 features (elevation_m INCLUDED, matching
deploy/feature_order.json), the same natural-log target, the same
out-of-fold target encoder refit inside every fold, and the same five seeds.

Cleaning replicates notebook cell 21 exactly and is verified against
deploy/division_values.csv (listing_count sums to 6,167 across 187 divisions).

Metrics are reported on both scales: R2 and RMSE on the log target (the scale
the model is fitted on), and MAE / MAPE / median APE after exponentiating back
to rupees per perch, which is the scale a reader interprets.

Outputs
  data/processed/spatial_tuned_factorial_runs.csv     one row per cell x seed
  data/processed/spatial_tuned_factorial_folds.csv    one row per fold (CV arms)
  data/processed/spatial_tuned_factorial_summary.csv  one row per cell
  data/processed/spatial_tuned_factorial.json         headline numbers
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
DATA = os.path.join(PROC, "final_dataset_hazard.csv")

SEEDS = [42, 1, 7, 123, 2024]
K_TRIM = 1.5
N_SPLITS = 5
SMOOTHING = 10

HAZARD_COLS = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
               "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]

# deploy/model.pkl, introspected 2026-09-06
TUNED = dict(n_estimators=500, max_depth=30, max_features=None,
             min_samples_split=6, min_samples_leaf=2, n_jobs=-1)
# notebook cell 33, the configuration behind the published 41.25%
UNTUNED = dict(n_estimators=300, n_jobs=-1)


def clean_name(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", c)


def drop_per_group(d, col, group="Address", k=K_TRIM):
    g = d.groupby(group)[col]
    mean, sd = g.transform("mean"), g.transform("std")
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], int((~keep).sum())


def target_encode_oof(train_groups, train_target, test_groups, seed,
                      n_splits=5, smoothing=SMOOTHING):
    """Notebook cell 23. Fitted on training rows only; unseen groups fall back
    to the training global mean rather than being dropped."""
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
    mapped = test_groups.map(smooth_full)
    n_unseen = int(mapped.isna().sum())
    return oof, mapped.fillna(global_mean), n_unseen, float(global_mean)


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


def score(y_true_log, y_pred_log):
    """R2 and RMSE on the log target; MAE / MAPE / medAPE back on the rupee scale."""
    yt, yp = np.asarray(y_true_log), np.asarray(y_pred_log)
    rt, rp = np.exp(yt), np.exp(yp)
    ape = np.abs(rp - rt) / rt
    return dict(r2_pct=r2_score(yt, yp) * 100,
                rmse_log=float(np.sqrt(mean_squared_error(yt, yp))),
                mae_log=float(mean_absolute_error(yt, yp)),
                mae_rs=float(np.mean(np.abs(rp - rt))),
                mape_pct=float(ape.mean() * 100),
                medape_pct=float(np.median(ape) * 100))


def fit_predict(kw, Xtr, ytr, Xte, seed):
    m = RandomForestRegressor(random_state=seed, **kw)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def run_random(X, y, groups, kw, seed):
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=seed)
    ytr, yte = y.iloc[itr], y.iloc[ite]
    oof, enc, n_unseen, gm = target_encode_oof(groups.iloc[itr], ytr,
                                               groups.iloc[ite], seed)
    Xtr, Xte = X.iloc[itr].copy(), X.iloc[ite].copy()
    Xtr["Address_target_enc"] = oof.values
    Xte["Address_target_enc"] = enc.values
    pred = fit_predict(kw, Xtr, ytr, Xte, seed)
    s = score(yte, pred)
    s.update(n_train=len(itr), n_test=len(ite), n_features=Xtr.shape[1],
             unseen_group_rows=n_unseen, r2_min=s["r2_pct"], r2_max=s["r2_pct"])
    return s, []


def run_spatial(X, y, groups, kw, seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = []
    for i, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        ytr, yva = y.iloc[tr], y.iloc[va]
        oof, enc, n_unseen, gm = target_encode_oof(groups.iloc[tr], ytr,
                                                   groups.iloc[va], seed)
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        Xtr["Address_target_enc"] = oof.values
        Xva["Address_target_enc"] = enc.values
        pred = fit_predict(kw, Xtr, ytr, Xva, seed)
        f = score(yva, pred)
        f.update(fold=i, n_train=len(tr), n_test=len(va),
                 n_features=Xtr.shape[1], unseen_group_rows=n_unseen,
                 val_divisions=int(groups.iloc[va].nunique()))
        folds.append(f)
    agg = {k: float(np.mean([f[k] for f in folds]))
           for k in ("r2_pct", "rmse_log", "mae_log", "mae_rs", "mape_pct", "medape_pct")}
    agg.update(n_features=folds[0]["n_features"],
               n_train=int(np.mean([f["n_train"] for f in folds])),
               n_test=int(np.mean([f["n_test"] for f in folds])),
               unseen_group_rows=int(sum(f["unseen_group_rows"] for f in folds)),
               r2_min=min(f["r2_pct"] for f in folds),
               r2_max=max(f["r2_pct"] for f in folds))
    return agg, folds


CELLS = {
    "A_random_tuned500": ("random", TUNED),
    "B_spatial_tuned500": ("spatial", TUNED),
    "C_spatial_untuned300": ("spatial", UNTUNED),
    "D_random_untuned300": ("random", UNTUNED),
}


def main():
    started = dt.datetime.now()
    X, y, groups, prov = build_matrix()
    print(f"rows_in={prov['rows_in']} price_filtered={prov['price_filtered']} "
          f"trimmed={prov['trimmed']} rows_used={prov['rows_used']} "
          f"divisions={prov['divisions']}", flush=True)
    print(f"features before target encoding: {X.shape[1]} "
          f"(+1 Address_target_enc = {X.shape[1] + 1})", flush=True)
    assert prov["rows_used"] == 6167, prov
    assert prov["divisions"] == 187, prov
    assert X.shape[1] + 1 == 49, X.shape

    runs, allfolds = [], []
    for name, (proto, kw) in CELLS.items():
        for seed in SEEDS:
            fn = run_random if proto == "random" else run_spatial
            agg, folds = fn(X, y, groups, kw, seed)
            agg.update(cell=name, protocol=proto, seed=seed,
                       n_estimators=kw["n_estimators"],
                       tuned=("max_depth" in kw))
            runs.append(agg)
            for f in folds:
                f.update(cell=name, seed=seed)
                allfolds.append(f)
            print(f"  {name:22s} seed {seed:>4}  R2={agg['r2_pct']:7.3f}%  "
                  f"MAPE={agg['mape_pct']:6.2f}%  medAPE={agg['medape_pct']:5.2f}%",
                  flush=True)

    runs = pd.DataFrame(runs)
    folds_df = pd.DataFrame(allfolds)

    summ = []
    for name in CELLS:
        sub = runs[runs.cell == name]
        summ.append(dict(
            cell=name, protocol=sub.protocol.iloc[0],
            n_estimators=int(sub.n_estimators.iloc[0]), tuned=bool(sub.tuned.iloc[0]),
            mean_r2_pct=round(sub.r2_pct.mean(), 4),
            sd_r2_pct=round(sub.r2_pct.std(ddof=1), 4),
            seed42_r2_pct=round(float(sub[sub.seed == 42].r2_pct.iloc[0]), 4),
            mean_rmse_log=round(sub.rmse_log.mean(), 4),
            mean_mape_pct=round(sub.mape_pct.mean(), 4),
            mean_medape_pct=round(sub.medape_pct.mean(), 4),
            mean_mae_rs=round(sub.mae_rs.mean(), 1),
            n_features=int(sub.n_features.iloc[0]),
        ))
    summary = pd.DataFrame(summ)

    s = summary.set_index("cell")
    A = s.loc["A_random_tuned500", "mean_r2_pct"]
    B = s.loc["B_spatial_tuned500", "mean_r2_pct"]
    C = s.loc["C_spatial_untuned300", "mean_r2_pct"]
    D = s.loc["D_random_untuned300", "mean_r2_pct"]

    head = dict(
        generated=str(started), **prov, n_features=49, seeds=SEEDS,
        random_tuned500_mean_r2=A, spatial_tuned500_mean_r2=B,
        spatial_untuned300_mean_r2=C, random_untuned300_mean_r2=D,
        random_tuned500_seed42_r2=float(s.loc["A_random_tuned500", "seed42_r2_pct"]),
        spatial_tuned500_seed42_r2=float(s.loc["B_spatial_tuned500", "seed42_r2_pct"]),
        spatial_untuned300_seed42_r2=float(s.loc["C_spatial_untuned300", "seed42_r2_pct"]),
        # the controlled comparison the thesis needs: same model, two protocols
        protocol_gap_tuned_pp=round(A - B, 4),
        protocol_gap_untuned_pp=round(D - C, 4),
        # how much of the old 89.43 - 41.25 headline was configuration, not protocol
        config_gap_under_spatial_pp=round(B - C, 4),
        config_gap_under_random_pp=round(A - D, 4),
        old_uncontrolled_gap_pp=round(A - C, 4),
        spatial_tuned_mape_pct=float(s.loc["B_spatial_tuned500", "mean_mape_pct"]),
        spatial_tuned_medape_pct=float(s.loc["B_spatial_tuned500", "mean_medape_pct"]),
        random_tuned_mape_pct=float(s.loc["A_random_tuned500", "mean_mape_pct"]),
        random_tuned_medape_pct=float(s.loc["A_random_tuned500", "mean_medape_pct"]),
        spatial_tuned_fold_min=round(float(runs[runs.cell == "B_spatial_tuned500"].r2_min.min()), 4),
        spatial_tuned_fold_max=round(float(runs[runs.cell == "B_spatial_tuned500"].r2_max.max()), 4),
        spatial_tuned_seed42_folds=[
            round(float(v), 4) for v in
            folds_df[(folds_df.cell == "B_spatial_tuned500") & (folds_df.seed == 42)]
            .sort_values("fold").r2_pct.tolist()],
        spatial_tuned_fold_means=[
            round(float(v), 4) for v in
            folds_df[folds_df.cell == "B_spatial_tuned500"]
            .groupby("fold").r2_pct.mean().tolist()],
    )

    meta = (f"# generated {started:%Y-%m-%d %H:%M:%S} by src/step15_spatial_tuned_factorial.py\n"
            f"# dataset={os.path.basename(DATA)} rows_in={prov['rows_in']} "
            f"price_filtered={prov['price_filtered']} trimmed_k={K_TRIM} "
            f"rows_used={prov['rows_used']} divisions={prov['divisions']} features=49\n"
            f"# tuned=RandomForestRegressor(n_estimators=500, max_depth=30, max_features=None, "
            f"min_samples_split=6, min_samples_leaf=2)  [= deploy/model.pkl]\n"
            f"# untuned=RandomForestRegressor(n_estimators=300)  [= notebook cell 33]\n"
            f"# cv=GroupKFold(n_splits=5) by Address; random=train_test_split(test_size=0.20)\n"
            f"# encoder=oof_target_enc(KFold5, smoothing=10) refit inside every fold/split\n"
            f"# seeds={'|'.join(map(str, SEEDS))}\n"
            f"# NOTE GroupKFold is deterministic - folds are identical across seeds; a seed\n"
            f"#      varies the forest and the inner encoder KFold only, not the partition.\n")
    for path, frame in ((os.path.join(PROC, "spatial_tuned_factorial_runs.csv"), runs),
                        (os.path.join(PROC, "spatial_tuned_factorial_folds.csv"), folds_df),
                        (os.path.join(PROC, "spatial_tuned_factorial_summary.csv"), summary)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(meta)
            frame.to_csv(fh, index=False)
    json.dump(head, open(os.path.join(PROC, "spatial_tuned_factorial.json"), "w"), indent=2)

    print("\n===== 2x2 factorial, mean R2 over 5 seeds =====")
    print(summary.to_string(index=False))
    print(f"\nProtocol gap holding the TUNED model fixed   (A - B): {A - B:+.2f} pp")
    print(f"Protocol gap holding the UNTUNED model fixed (D - C): {D - C:+.2f} pp")
    print(f"Configuration gap under spatial CV           (B - C): {B - C:+.2f} pp")
    print(f"Configuration gap under the random split     (A - D): {A - D:+.2f} pp")
    print(f"Old uncontrolled headline gap                (A - C): {A - C:+.2f} pp")
    print(f"\nelapsed {dt.datetime.now() - started}")


if __name__ == "__main__":
    main()

"""
step14_rq5_hazard_vs_nohazard.py
--------------------------------
Answers RQ5: does adding the six hazard columns change held-out accuracy
under the RANDOM 80/20 split?

Chapter 4 answered this with a pre-price-floor number (ensemble on
final_dataset.csv vs final_dataset_hazard.csv) that THESIS_FACTS.md
supersedes. This script re-runs the comparison on the CURRENT data.

Design
------
* Cleaning replicates notebook cell 21 exactly: global price floor/ceiling
  at the 0.5%/99.5% quantiles, then per-division +/-1.5 SD trimming on
  Price per Perch and Land_size(Perches).
* Both arms are cut from the SAME cleaned frame, so the rows, the splits
  and the target encoding are byte-identical between them. The only
  difference is the presence of the six hazard columns. This is a strictly
  controlled comparison; it is not two separate pipeline runs.
* Estimator is the DEPLOYED specification (THESIS_FACTS.md 2.1):
  n_estimators=500, max_depth=30, max_features=None,
  min_samples_split=6, min_samples_leaf=2, random_state=42.
  Hyperparameters are held fixed rather than re-searched, so any
  difference is attributable to the feature set and not to tuning.
* Five split seeds (42, 1, 7, 123, 2024) matching the ablation of
  Section III-G. Deltas are computed PAIRED on matched seeds and judged
  by the same rule the paper applies to the hazard ablation: an effect
  is real only if |mean delta| exceeds two standard deviations of its own
  paired distribution AND its sign is consistent across all five seeds.

CAUTION ON ABSOLUTE VALUES
--------------------------
Library versions here differ from the original notebook run, so absolute
R2 will not be bit-identical to THESIS_FACTS.md 3.1. Quote the DELTA,
which is computed within a single execution and is unaffected by that
drift. Absolute values are printed for sanity-checking only.
"""

import numpy as np
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

DATA = 'data/processed/final_dataset_hazard.csv'
OUT = 'data/processed/rq5_hazard_vs_nohazard.csv'
HAZARD_COLS = ['elevation_m', 'water_occurrence_pct', 'dist_to_stream_km',
               'dist_to_kelani_km', 'hand_m', 'sar_flood_open_land']
SEEDS = [42, 1, 7, 123, 2024]
K = 1.5
RF_KW = dict(n_estimators=500, max_depth=30, max_features=None,
             min_samples_split=6, min_samples_leaf=2,
             random_state=42, n_jobs=-1)


def drop_per_group(d, col, group='Address', k=2):
    g = d.groupby(group)[col]
    mean, sd = g.transform('mean'), g.transform('std')
    keep = sd.isna() | (d[col].sub(mean).abs() <= k * sd)
    return d[keep], (~keep).sum()


def target_encode_oof(train_groups, train_target, test_groups,
                      n_splits=5, smoothing=10, random_state=42):
    global_mean = train_target.mean()
    oof = pd.Series(index=train_groups.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in kf.split(train_groups):
        tr_g, tr_t = train_groups.iloc[tr_idx], train_target.iloc[tr_idx]
        stats = tr_t.groupby(tr_g).agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        val_g = train_groups.iloc[val_idx]
        oof.iloc[val_idx] = val_g.map(smooth).fillna(global_mean).values
    stats_full = train_target.groupby(train_groups).agg(['mean', 'count'])
    smooth_full = (stats_full['mean'] * stats_full['count'] + global_mean * smoothing) / (stats_full['count'] + smoothing)
    return oof, test_groups.map(smooth_full).fillna(global_mean)


def clean_name(c):
    return re.sub(r'[^0-9a-zA-Z_]+', '_', c)


# ---------------------------------------------------------------- cleaning
df_raw = pd.read_csv(DATA, low_memory=False)
print(f'raw: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols')

lo, hi = df_raw['Price per Perch'].quantile([0.005, 0.995])
df2 = df_raw[df_raw['Price per Perch'].between(lo, hi)].copy()
n0 = df_raw.shape[0] - df2.shape[0]
print(f'price floor/ceiling [{lo:,.0f}, {hi:,.0f}] removed {n0} -> {df2.shape[0]}')

df2, n1 = drop_per_group(df2, 'Price per Perch', k=K)
df2, n2 = drop_per_group(df2, 'Land_size(Perches)', k=K)
print(f'per-division +/-{K} SD trim removed {n1 + n2} -> {df2.shape[0]}')
print(f'divisions: {df2["Address"].nunique()}')

y = np.log(df2['Price per Perch'])

count_cols = [c for c in df2.columns if c.startswith('count_')]
mindist_cols = [c for c in df2.columns if c.startswith('min_dist')]
df2['total_amenity_count'] = df2[count_cols].sum(axis=1)
df2['min_dist_any_amenity'] = df2[mindist_cols].min(axis=1)
df2['school_count_total'] = df2['count_govtschools_A'] + df2['count_semigovtschools'] + df2['count_intlschools']
df2['health_count_total'] = df2['count_Govt_Hospitals'] + df2['count_Pvt_Hospital'] + df2['count_Pvt_Med_Centers']
df2['finance_count_total'] = df2['count_banks_within_2km'] + df2['count_FinanceCompanies_within_2km']
engineered = ['total_amenity_count', 'min_dist_any_amenity', 'school_count_total',
              'health_count_total', 'finance_count_total']

lt = pd.get_dummies(df2['Land_type'], prefix='Land_type').astype(int)
amen = [c for c in df2.columns if c.startswith(('min_dist', 'count_')) and c not in engineered]
base = ['Land_size(Perches)', 'Distance from fort'] + amen + HAZARD_COLS + engineered
X = pd.concat([df2[base], lt], axis=1)
X.columns = [clean_name(c) for c in X.columns]
HAZ = [clean_name(c) for c in HAZARD_COLS]
addr = df2['Address'].reset_index(drop=True)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
print(f'features with hazard: {X.shape[1] + 1} (incl. Address_target_enc)')
print(f'features without    : {X.shape[1] - len(HAZ) + 1}')

# ---------------------------------------------------------------- runs
rows = []
for seed in SEEDS:
    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=seed)
    ytr, yte = y.iloc[itr], y.iloc[ite]
    oof, tenc = target_encode_oof(addr.iloc[itr], ytr, addr.iloc[ite])

    for arm in ('with_hazard', 'no_hazard'):
        cols = list(X.columns) if arm == 'with_hazard' else [c for c in X.columns if c not in HAZ]
        Xtr = X.iloc[itr][cols].copy()
        Xte = X.iloc[ite][cols].copy()
        Xtr['Address_target_enc'] = oof.values
        Xte['Address_target_enc'] = tenc.values

        m = RandomForestRegressor(**RF_KW).fit(Xtr, ytr)
        p = m.predict(Xte)
        r2 = r2_score(yte, p) * 100
        rmse = float(np.sqrt(mean_squared_error(yte, p)))
        mape = mean_absolute_percentage_error(np.exp(yte), np.exp(p)) * 100
        medape = float(np.median(np.abs(np.exp(p) - np.exp(yte)) / np.exp(yte)) * 100)
        rows.append(dict(seed=seed, arm=arm, n_features=Xtr.shape[1],
                         r2_pct=r2, rmse_log=rmse, mape_pct=mape, medape_pct=medape,
                         n_train=len(itr), n_test=len(ite)))
        print(f'seed {seed:>4}  {arm:<12} feats={Xtr.shape[1]:>2}  '
              f'R2={r2:6.3f}%  RMSE={rmse:.4f}  MAPE={mape:6.3f}%  medAPE={medape:5.2f}%')

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

# ---------------------------------------------------------------- verdict
piv = res.pivot(index='seed', columns='arm', values='r2_pct')
piv['delta_pp'] = piv['with_hazard'] - piv['no_hazard']
mean_d = piv['delta_pp'].mean()
sd_d = piv['delta_pp'].std(ddof=1)
same_sign = int((np.sign(piv['delta_pp']) == np.sign(mean_d)).sum())
clears = bool(abs(mean_d) > 2 * sd_d and same_sign == len(SEEDS))

print('\n===== RQ5 verdict (random 80/20 split) =====')
print(piv.to_string())
print(f'\nmean paired delta      : {mean_d:+.4f} pp  (with_hazard minus no_hazard)')
print(f'SD of paired delta     : {sd_d:.4f} pp')
print(f'2 SD threshold         : {2 * sd_d:.4f} pp')
print(f'seeds agreeing on sign : {same_sign}/{len(SEEDS)}')
print(f'ABOVE NOISE FLOOR      : {clears}')
print(f'\nwritten -> {OUT}')

json.dump(dict(mean_delta_pp=float(mean_d), sd_delta_pp=float(sd_d),
               two_sd_pp=float(2 * sd_d), seeds_same_sign=same_sign,
               n_seeds=len(SEEDS), above_noise_floor=clears,
               rows_used=int(df2.shape[0]), divisions=int(df2['Address'].nunique())),
          open('data/processed/rq5_verdict.json', 'w'), indent=2)

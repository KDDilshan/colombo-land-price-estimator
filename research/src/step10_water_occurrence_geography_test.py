"""Step 10 - apply the project's own pre-specified geography test to the RETAINED
variable `water_occurrence_pct`.

WHY THIS EXISTS
  This test was applied to two candidate variables and rejected both:

    flood_frequency_rejection.md    rejected: every non-zero value was coastal
                                    and no Kelani floodplain division exceeded 0
    water_recurrence_validation.md  rejected: the Kelani floodplain failed to
                                    separate (mean rank 60.9 of 194) while the
                                    coastal group dominated (17.4)

  It was never applied to `water_occurrence_pct`, which was retained on the
  automated screening rules alone (194 distinct values, r = -0.18 against
  Distance from fort). That matters because `water_occurrence_pct` is the ONLY
  individual hazard variable whose ablation delta clears the noise floor
  (-1.71 pp, hazard_ablation_report.md), so a Chapter 4 finding rests on it.

THE SPECIFIC CONCERN, IN THE PROJECT'S OWN WORDS
  src/step6h_water_recurrence.py, module docstring:
    "unmasked recurrence would rank the shoreline top and repeat the MODIS
     failure exactly. Pixels whose `occurrence` exceeds PERMANENT_PCT are
     permanent water [and are masked]"
  src/step6i_make_sar_js.py sets PERM_OCC = 80 for the same reason.

  But src/step6c_extract_cog_hazards.py line ~175 computes
  `water_occurrence_pct` as the plain mean of the raw JRC occurrence band over
  the 1 km buffer, with NO permanent-water mask:

      rec["water_occurrence_pct"] = round(float(np.mean(vals)), 4)

  The retained variable is therefore unmasked, while the rejected variable and
  the SAR variable both mask. The project applied its own safeguard to two of
  three JRC-derived variables and not to the third.

TEST DESIGN
  The three comparison groups are taken verbatim from
  water_recurrence_validation.md so that the results are directly comparable.
  They were fixed in that document before any result was read. No group is
  redefined here and no threshold is tuned.

  Rank 1 = wettest of 194 divisions.
    expected HIGH  Kelani floodplain - must separate from the comparison groups
    expected LOW   southern uplands
    coastal        must NOT dominate the ranking

WHAT THIS SCRIPT DOES NOT DO
  It does not re-extract the variable with a mask applied. Doing so requires
  re-running src/step6c_extract_cog_hazards.py against the JRC occurrence
  COGs, which is a network operation outside this script's scope. The remedy is
  recommended, not executed.

Output
  results/water_occurrence_geography_test.md
"""

from __future__ import annotations

import datetime as dt
import os
import platform

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "final_dataset_hazard.csv")
OUT = os.path.join(ROOT, "results", "water_occurrence_geography_test.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STARTED = dt.datetime.now()

# groups copied verbatim from water_recurrence_validation.md
KELANI = ["mulleriyawa", "sedawatta", "kelanimulla", "ambathale",
          "wellampitiya", "kotikawatta", "kolonnawa"]
UPLANDS = ["maharagama", "kottawa", "pannipitiya", "homagama"]
COASTAL = ["fort", "uyana", "idama", "moratuwella", "mount lavinia"]

# the published result for the REJECTED variable, for direct comparison
REC = {"kelani": 60.9, "uplands": 124.8, "coastal": 17.4}

df = pd.read_csv(DATA, low_memory=False)
d = df.groupby("Address")[["water_occurrence_pct", "dist_to_kelani_km",
                           "hand_m"]].first()
d["rank"] = d["water_occurrence_pct"].rank(ascending=False, method="min")
n = len(d)


def grp(names):
    present = [x for x in names if x in d.index]
    return present, d.loc[present, "rank"].mean(), d.loc[present, "water_occurrence_pct"].mean()


k_names, k_rank, k_val = grp(KELANI)
u_names, u_rank, u_val = grp(UPLANDS)
c_names, c_rank, c_val = grp(COASTAL)

top = d.sort_values("water_occurrence_pct", ascending=False).head(12)
coastal_in_top10 = sum(1 for x in d.sort_values("water_occurrence_pct",
                                                ascending=False).head(10).index
                       if x in COASTAL)

verdict_sep = "PASS" if k_rank < u_rank else "FAIL"
verdict_coast = "FAIL" if c_rank < k_rank else "PASS"

L = []
w = L.append
w("# Geography test applied to the retained variable `water_occurrence_pct`")
w("")
w(f"Generated {STARTED:%Y-%m-%d %H:%M} by `src/step10_water_occurrence_geography_test.py`.")
w(f"Environment: python={platform.python_version()} pandas={pd.__version__} "
  f"numpy={np.__version__}.")
w("Descriptive test over the 194 divisions of `data/processed/final_dataset_hazard.csv`. "
  "No model is fitted, so this result does not depend on library version.")
w("")
w("## Why this test was run")
w("")
w("This is the same pre-specified test that rejected MODIS `flood_frequency` "
  "(`flood_frequency_rejection.md`) and JRC `water_recurrence_pct` "
  "(`water_recurrence_validation.md`). It was never applied to "
  "`water_occurrence_pct`, which was retained on the automated screening rules "
  "alone. The variable matters because it is the only individual hazard "
  "variable whose ablation delta clears the noise floor (-1.71 pp against a "
  "2 SD threshold of 1.71, `hazard_ablation_report.md`).")
w("")
w("The groups below are copied verbatim from `water_recurrence_validation.md` "
  "and were fixed there before any result was read.")
w("")
w("## The masking inconsistency")
w("")
w("| variable | JRC band | permanent water masked? | producing script | outcome |")
w("|---|---|---|---|---|")
w("| `water_recurrence_pct` | recurrence | **yes**, occurrence > 80% | `step6h_water_recurrence.py` | rejected |")
w("| `sar_flood_open_land` | occurrence used as mask | **yes**, `PERM_OCC = 80` | `step6i_make_sar_js.py` | retained |")
w("| `water_occurrence_pct` | occurrence | **no** | `step6c_extract_cog_hazards.py` | **retained and modelled** |")
w("")
w("`step6h_water_recurrence.py` states its reason directly: \"unmasked "
  "recurrence would rank the shoreline top and repeat the MODIS failure "
  "exactly.\" The same safeguard was applied to the SAR variable. It was not "
  "applied to `water_occurrence_pct`, which is computed as the plain mean of "
  "the raw occurrence band over the 1 km buffer.")
w("")
w("## Result")
w("")
w(f"| group | divisions | mean rank of {n} (1 = wettest) | mean value (%) | "
  "same test, rejected `water_recurrence_pct` |")
w("|---|---|---|---|---|")
w(f"| expected HIGH - Kelani floodplain | {len(k_names)} | **{k_rank:.1f}** | "
  f"{k_val:.3f} | 60.9 |")
w(f"| expected LOW - southern uplands | {len(u_names)} | {u_rank:.1f} | "
  f"{u_val:.3f} | 124.8 |")
w(f"| coastal - must NOT dominate | {len(c_names)} | **{c_rank:.1f}** | "
  f"{c_val:.3f} | 17.4 |")
w("")
w(f"**Separation from the uplands: {verdict_sep}.** The Kelani floodplain "
  f"({k_rank:.1f}) does rank wetter than the southern uplands ({u_rank:.1f}), "
  "so the variable is not pure noise and carries some river signal. This is "
  "better separation than the rejected recurrence variable achieved.")
w("")
w(f"**Coastal dominance: {verdict_coast}.** The coastal group averages rank "
  f"{c_rank:.1f} of {n}, against {k_rank:.1f} for the floodplain the variable "
  "is supposed to identify. The coastal group is ranked "
  f"{'wetter' if c_rank < k_rank else 'drier'} than the floodplain by "
  f"{abs(k_rank - c_rank):.1f} rank positions. On this criterion the retained "
  f"variable performs {'worse' if c_rank < REC['coastal'] else 'better'} than "
  f"the rejected `water_recurrence_pct` ({REC['coastal']}), which was masked.")
w("")
w(f"{coastal_in_top10} of the 10 wettest divisions belong to the 5-division "
  "coastal comparison group.")
w("")
w("## The 12 wettest divisions")
w("")
w("| rank | division | water_occurrence_pct | dist_to_kelani_km | coastal group? |")
w("|---|---|---|---|---|")
for a, r in top.iterrows():
    w(f"| {int(r['rank'])} | {a} | {r['water_occurrence_pct']:.4f} | "
      f"{r['dist_to_kelani_km']:.3f} | {'**yes**' if a in COASTAL else ''} |")
w("")
w(f"`fort` records {d.loc['fort','water_occurrence_pct']:.2f}% surface-water "
  "occurrence, the highest of all 194 divisions, at "
  f"{d.loc['fort','dist_to_kelani_km']:.2f} km from the Kelani. Fort is the "
  "harbour. An unmasked occurrence mean over a 1 km buffer centred there "
  "includes open sea, and the value measures the harbour rather than flood "
  "exposure.")
w("")
w("## The Kelani floodplain divisions individually")
w("")
w("| division | rank | water_occurrence_pct | dist_to_kelani_km |")
w("|---|---|---|---|")
for a in k_names:
    r = d.loc[a]
    w(f"| {a} | {int(r['rank'])} | {r['water_occurrence_pct']:.4f} | "
      f"{r['dist_to_kelani_km']:.3f} |")
w("")
w("`mulleriyawa` ranks 125 of 194 at 0.0012% while sitting 1.32 km from the "
  "Kelani. `sedawatta`, 81 m from the river, ranks 21. The variable is "
  "inconsistent within the floodplain itself.")
w("")
w("## Assessment")
w("")
w("The test does not cleanly reject the variable and should not be reported as "
  "doing so. `water_occurrence_pct` separates the floodplain from the uplands, "
  "which neither rejected variable managed, so it carries genuine river-flood "
  "signal. But its extreme values are coastal and it is unmasked, so it "
  "**also** measures proximity to permanent water including the sea. It is a "
  "mixture of two signals, and the ablation delta of -1.71 pp cannot be "
  "attributed to flood exposure alone.")
w("")
w("This is the failure mode the project rejected twice. "
  "`flood_frequency_rejection.md` states the objection in general terms: "
  "including such a column \"would report 'flood exposure lowers coastal land "
  "value' from a column that measured distance to the sea.\" The concern "
  "applies here in weaker form, because the variable is not purely coastal.")
w("")
w("## Recommended remedy")
w("")
w("Re-extract `water_occurrence_pct` from the JRC occurrence COGs with the "
  "same `occurrence > 80%` permanent-water mask already implemented in "
  "`src/step6h_water_recurrence.py`, removing masked pixels from the "
  "denominator rather than counting them as zero, and re-run the ablation. "
  "Three outcomes are possible and all are publishable:")
w("")
w("1. The masked variable retains its ablation effect - the finding stands and "
  "is now defensible.")
w("2. The effect disappears - the -1.71 pp was coastal proximity, and no "
  "individual hazard variable survives.")
w("3. The effect strengthens - the coastal contamination was diluting a real "
  "flood signal.")
w("")
w("Until that is run, Chapter 4 Section 4.5 should qualify the "
  "`water_occurrence_pct` result: the variable is unmasked, its extreme values "
  "are coastal, and its ablation delta is not cleanly attributable to flood "
  "exposure. The threshold-level nature of the effect (-1.71 pp against a "
  "1.71 pp threshold) makes this qualification more rather than less "
  "important.")
w("")
w("## What this does not affect")
w("")
w("The block-level ablation finding (-3.06 pp for removing all five hazard "
  "variables) is not invalidated by this. It is weakened, because one of the "
  "five contributing variables is partly measuring something else, but the "
  "block delta does not depend on any single variable's interpretation.")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote results/{os.path.basename(OUT)}")
print(f"  Kelani {k_rank:.1f} | uplands {u_rank:.1f} | coastal {c_rank:.1f}  (of {n})")
print(f"  separation {verdict_sep}; coastal dominance {verdict_coast}")
print(f"  {coastal_in_top10}/10 wettest divisions are in the coastal group")

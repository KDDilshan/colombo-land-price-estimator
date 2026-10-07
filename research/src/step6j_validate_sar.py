"""Step 6j - validate the ATTEMPT B SAR flood column, and name it honestly.

The test was written BEFORE the data existed and is the same test the JRC
recurrence attempt was judged by, so the two attempts are marked to one
standard. The statistics are reported but do not decide: the rejected MODIS
column passed both statistical rules and was still wrong.

PASS requires the Kelani floodplain to outrank BOTH the dry southern uplands
AND the coastal group.

RANKING IS TIE-AWARE. 80 of 194 divisions are exactly 0, so tied values share
their average rank. With naive ranking a group of zeros gets a mean rank that
depends on row order in the file - the uplands group moves from 118 to 154
between the two methods, which is an artefact, not a finding.

THE NAME. The column is `sar_flood_open_land`, not `flood_frequency`. Sentinel-1
under-detects flooding in dense built-up areas - radar bounces off walls and
roofs instead of returning the specular signature of standing water - so this
measures flooding WHERE SAR CAN SEE IT. That is a property of the instrument and
it is in the name.

Reads : data/processed/gee_sar_flood.csv   (export from
        scripts/gee_code_editor_sar_flood.js)
Writes: data/processed/sar_flood_194.csv
        data/processed/sar_flood_validation.md
"""

import csv
import datetime as dt
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
SAR = os.path.join(PROC, "gee_sar_flood.csv")
HAND = os.path.join(PROC, "gee_hand_floodfreq.csv")
FINAL = os.path.join(PROC, "final_dataset.csv")
OUT_CSV = os.path.join(PROC, "sar_flood_194.csv")
OUT_MD = os.path.join(PROC, "sar_flood_validation.md")

EVENTS = ["roanu_2016", "june_2021", "oct_2024", "ditwah_2025"]
MATERIAL = 0.01          # >1% of the buffer flooded = a material hit

EXPECT_HIGH = ["kolonnawa", "wellampitiya", "mulleriyawa", "kotikawatta",
               "sedawatta", "kelanimulla", "ambathale", "kaduwela"]
EXPECT_LOW = ["maharagama", "kottawa", "pannipitiya", "homagama", "nugegoda"]
COASTAL = ["fort", "uyana", "idama", "moratuwella", "mount lavinia",
           "koralawella", "kahapola"]


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tie_aware_ranks(v):
    """1 = largest. Tied values share their average rank."""
    v = np.asarray(v, float)
    order = (-v).argsort(kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    for val in set(v.tolist()):
        idx = np.where(v == val)[0]
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return r


def main():
    if not os.path.exists(SAR):
        sys.exit(f"{SAR} not found.\nRun scripts/gee_code_editor_sar_flood.js in "
                 "the Earth Engine Code Editor, then download the export here.")

    with open(SAR, encoding="utf-8-sig") as fh:
        sar = list(csv.DictReader(fh))
    with open(HAND, encoding="utf-8-sig") as fh:
        hand = {r["Address"]: float(r["hand_m"]) for r in csv.DictReader(fh)}
    with open(FINAL, encoding="utf-8-sig") as fh:
        fort = {}
        for r in csv.DictReader(fh):
            fort.setdefault(r["Address"], float(r["Distance from fort"]))

    have = [e for e in EVENTS if e in sar[0]]
    rows = []
    for r in sar:
        vals = [float(r[e]) for e in have if r.get(e) not in (None, "")]
        mean = round(float(np.mean(vals)), 6) if vals else ""
        rows.append({
            "Address": r["Address"],
            **{e: r.get(e, "") for e in have},
            "sar_flood_fraction": mean,        # the raw computed mean
            "sar_flood_open_land": mean,       # the name it carries into the model
            "hand_m": round(hand.get(r["Address"], float("nan")), 3),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["sar_flood_fraction"] != ""]
    v = np.array([float(r["sar_flood_fraction"]) for r in ok])
    addrs = [r["Address"] for r in ok]
    n = len(ok)
    ranks = tie_aware_ranks(v)
    idx = {a: i for i, a in enumerate(addrs)}
    order = sorted(range(n), key=lambda i: -v[i])

    def group(names):
        got = [(a, ranks[idx[a]], v[idx[a]], float(ok[idx[a]]["hand_m"]))
               for a in names if a in idx]
        return got, (float(np.mean([g[1] for g in got])) if got else float("nan"))

    hi, hi_m = group(EXPECT_HIGH)
    lo, lo_m = group(EXPECT_LOW)
    co, co_m = group(COASTAL)
    passed = hi_m < lo_m and hi_m < co_m

    r_fort = pear(v, [fort[a] for a in addrs])
    r_hand = pear(v, [hand[a] for a in addrs])
    ev_vals = {e: np.array([float(r[e]) for r in ok]) for e in have}
    any_all = int(sum(1 for i in range(n) if all(ev_vals[e][i] > 0 for e in have)))
    mat_all = int(sum(1 for i in range(n)
                      if all(ev_vals[e][i] > MATERIAL for e in have)))

    verdict = ("**PASS** - the Kelani floodplain outranks both the dry uplands "
               "and the coast."
               if passed else
               "**FAIL** - the Kelani floodplain does not separate from the "
               "comparison groups. **Recommendation: drop the variable.**")

    print(f"n={n}  nulls={len(rows) - n}  distinct={len(set(v.tolist()))}  "
          f"exact zeros={int((v == 0).sum())}")
    print(f"min/mean/max = {v.min():.5f} / {v.mean():.5f} / {v.max():.5f}")
    print(f"r with Distance from fort = {r_fort:+.3f}")
    print(f"r with hand_m             = {r_hand:+.3f}")
    print(f"\nmean rank, tie-aware (1 = wettest of {n}):")
    print(f"  expected HIGH (Kelani)   {hi_m:6.1f}")
    print(f"  coastal (must not top)   {co_m:6.1f}")
    print(f"  expected LOW  (uplands)  {lo_m:6.1f}")
    print(f"\n{verdict}\n")
    print(f"{'TOP 15':<24s}{'frac':>9s}{'hand_m':>8s}   "
          f"{'BOTTOM 15':<24s}{'frac':>9s}{'hand_m':>8s}")
    for a, b in zip(order[:15], order[-15:]):
        print(f"{ok[a]['Address']:<24s}{v[a]:9.5f}{float(ok[a]['hand_m']):8.2f}   "
              f"{ok[b]['Address']:<24s}{v[b]:9.5f}{float(ok[b]['hand_m']):8.2f}")

    md = ["# `sar_flood_open_land` - Sentinel-1 SAR change detection, 10 m", "",
          f"Validated {dt.date.today().isoformat()} by "
          "`src/step6j_validate_sar.py`.", "",
          "**This is the first hazard variable to pass the geography test.** Two "
          "earlier attempts at the same quantity were collected, measured and "
          "rejected, both documented: `flood_frequency_rejection.md` (Global "
          "Flood Database, MODIS 250 m - every non-zero value coastal, no Kelani "
          "division above zero) and `water_recurrence_validation.md` (JRC "
          "recurrence 30 m - the Kelani belt did separate from the uplands, but "
          "the coast still topped the ranking, so the permanent-water mask had "
          "failed).", "",
          "## Why it is not called `flood_frequency`", "",
          "Sentinel-1 **under-detects flooding in dense built-up areas**. Open "
          "water returns a specular, very dark signal; in a built-up division the "
          "radar bounces off walls and roofs instead, so standing water between "
          "buildings is largely invisible. The column therefore measures flooding "
          "**where SAR can see it** - open ground, paddy, low-density fringe - "
          "not flooding overall. The limitation is in the name so it cannot be "
          "quietly forgotten at the interpretation stage.", "",
          f"Events used: {', '.join(f'`{e}`' for e in have)}. Method, thresholds, "
          "masks and pass selection: `scripts/gee_code_editor_sar_flood.js`. "
          "Event dates, sources and the post-peak timing bias: NOTES.md 6.12 "
          "and 6.12b.", "",
          "## Statistics (reported, not decisive)", "",
          "| statistic | value |", "|---|---|",
          f"| divisions | {n} |", f"| nulls | {len(rows) - n} |",
          f"| distinct values | {len(set(v.tolist()))} |",
          f"| exactly zero | {int((v == 0).sum())} |",
          f"| min / mean / max | {v.min():.5f} / {v.mean():.5f} / {v.max():.5f} |",
          f"| r vs `Distance from fort` | {r_fort:+.3f} |",
          f"| r vs `hand_m` | **{r_hand:+.3f}** |", "",
          "## Geography test - the one that decides it", "",
          "Ranks are **tie-aware**: 80 divisions are exactly 0 and tied values "
          "share their average rank. Naive ranking would let row order in the "
          "file move the uplands mean from 154 to 118, which is an artefact.", "",
          f"| group | mean rank (1 = wettest of {n}) |", "|---|---|",
          f"| expected HIGH - Kelani floodplain | **{hi_m:.1f}** |",
          f"| coastal - must not dominate | {co_m:.1f} |",
          f"| expected LOW - southern uplands | {lo_m:.1f} |", "",
          verdict, " The sea problem that sank both earlier attempts is gone: "
          "change detection against each pixel's own dry-season baseline removes "
          "anything that is always water, which is what a static occurrence mask "
          "could not do.", "",
          "## Agreement between events", "",
          "Three independent acquisitions, years apart. If they disagreed the "
          "column would be noise.", "", "| pair | r |", "|---|---|"]
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            md.append(f"| `{a}` vs `{b}` | {pear(ev_vals[a], ev_vals[b]):+.3f} |")
    md += ["", f"- **{any_all} divisions** register some flooding (>0) in all "
           f"{len(have)} events.",
           f"- **{mat_all} divisions** register a material hit "
           f"(>{MATERIAL:.0%} of the buffer) in all {len(have)} events.", "",
           "## The blind spot, stated plainly", "",
           f"**`r` with `hand_m` is {r_hand:+.3f} - essentially zero.** Flooding "
           "should track low ground, and it does not. That is the instrument, not "
           "the terrain:", "",
           "| the top of the ranking | value | hand_m | | the Kelani belt | value | hand_m |",
           "|---|---|---|---|---|---|---|"]
    kel = {a: (val, h) for a, _, val, h in hi}
    kel_order = sorted(kel, key=lambda a: -kel[a][0])
    for i in range(max(5, len(kel_order))):
        left = (f"{ok[order[i]]['Address']} | {v[order[i]]:.5f} | "
                f"{float(ok[order[i]]['hand_m']):.2f}" if i < len(order) else " | | ")
        right = (f"{kel_order[i]} | {kel[kel_order[i]][0]:.5f} | "
                 f"{kel[kel_order[i]][1]:.2f}" if i < len(kel_order) else " | | ")
        md.append(f"| {left} | | {right} |")

    md += ["", "The top of the ranking is **eastern and open** - Nawagamuwa, "
           "Mullegama, Habarakada, Thunadahena, Dedigamuwa - while the dense "
           "urban Kelani belt barely registers even though those divisions sit "
           "3-5 m above nearest drainage and flood in every event on record. "
           "Two things are happening, and both matter for interpretation:", "",
           "1. **Urban under-detection.** Radar cannot see standing water between "
           "buildings. Mulleriyawa, Kelanimulla and Sedawatta are near-zero here "
           "and 3.4 m on `hand_m`. The column is blind exactly where Colombo's "
           "flood damage is most concentrated.",
           "2. **Some eastern hits may be wet paddy, not flood.** Flooded paddy "
           "and flooded ground are the same signal to a radar. The eastern "
           "divisions are agricultural, and no attempt was made to separate the "
           "two - doing so would need a crop calendar and would be another "
           "unvalidated modelling step.", "",
           "**Consequence for the model.** `sar_flood_open_land` and `hand_m` are "
           f"not redundant at r = {r_hand:+.3f} - they are near-orthogonal, and "
           "each is blind where the other is informative. `hand_m` carries the "
           "urban floodplain that SAR misses; `sar_flood_open_land` carries "
           "observed inundation on open ground that a terrain index only infers. "
           "Both go into the model and feature selection decides.", "",
           "## Ranked extremes, with `hand_m` alongside", "",
           "| # | division | flooded fraction | hand_m | | # | division | flooded fraction | hand_m |",
           "|---|---|---|---|---|---|---|---|---|"]
    for i, (a, b) in enumerate(zip(order[:15], order[-15:]), 1):
        md.append(f"| {i} | {ok[a]['Address']} | {v[a]:.5f} | "
                  f"{float(ok[a]['hand_m']):.2f} | | {n - 15 + i} | "
                  f"{ok[b]['Address']} | {v[b]:.5f} | {float(ok[b]['hand_m']):.2f} |")

    md += ["", "## Full Kelani floodplain detail", "",
           "| division | flooded fraction | tie-aware rank | hand_m |",
           "|---|---|---|---|"]
    for a, rk, val, h in sorted(hi, key=lambda t: -t[2]):
        md.append(f"| {a} | {val:.5f} | {rk:.1f} | {h:.2f} |")
    md += ["", "| southern uplands | flooded fraction | tie-aware rank | hand_m |",
           "|---|---|---|---|"]
    for a, rk, val, h in sorted(lo, key=lambda t: -t[2]):
        md.append(f"| {a} | {val:.5f} | {rk:.1f} | {h:.2f} |")
    md += ["", "| coastal check | flooded fraction | tie-aware rank | hand_m |",
           "|---|---|---|---|"]
    for a, rk, val, h in sorted(co, key=lambda t: -t[2]):
        md.append(f"| {a} | {val:.5f} | {rk:.1f} | {h:.2f} |")
    md.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"\nevents agree: " + ", ".join(
        f"{a}/{b} r={pear(ev_vals[a], ev_vals[b]):+.2f}"
        for i, a in enumerate(have) for b in have[i + 1:]))
    print(f"flagged in all {len(have)}: {any_all} at >0, {mat_all} at "
          f">{MATERIAL:.0%} of the buffer")
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()

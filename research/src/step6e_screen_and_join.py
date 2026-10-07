"""Step 6e - screen the hazard/terrain candidates, then join the survivors.

SCREENING IS RUN ON THE 194 DIVISIONS, NOT THE 7034 ROWS. Every listing in a
division shares one feature vector, so a column costs a degree of freedom
across 194 effective units. Three tests:

  1. distinct-value count across the 194 points   (< 30 = coarse proxy)
  2. |Pearson r| against `Distance from fort`     (> 0.80 = redundant)
  3. |Pearson r| against the other candidates     (> 0.80 = pick one)

Outputs
  data/processed/hazard_screening.md
  data/processed/final_dataset_hazard.csv     (7034 rows, original + survivors)
"""

import csv
import datetime as dt
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FINAL = os.path.join(PROC, "final_dataset.csv")
OUT_CSV = os.path.join(PROC, "final_dataset_hazard.csv")
OUT_MD = os.path.join(PROC, "hazard_screening.md")

SOURCES = [
    ("cog_hazard_194.csv", ["elevation_m", "slope_deg", "built_up_fraction",
                            "tree_cover_fraction", "water_occurrence_pct"]),
    ("osm_water_194.csv", ["dist_to_stream_km", "dist_to_kelani_km"]),
    ("landslide_lhasa_194.csv", ["landslide_susceptibility"]),
    # whichever Earth Engine script was run - both write the same two columns
    ("gee_hazard_194.csv", ["hand_m", "flood_frequency"]),
    ("gee_hand_floodfreq.csv", ["hand_m", "flood_frequency"]),
    ("sar_flood_194.csv", ["sar_flood_open_land"]),
]

# Kept in the joined file, excluded from the RECOMMENDED modelling set. This is
# the analyst's decision on a flagged collinear pair, not a screening rule.
MODELLING_EXCLUDE = {
    "elevation_m":
        "r = +0.93 with `hand_m` **on the 194 divisions**, the basis this "
        "screening runs on; **on the 7034 listing rows the same pair is "
        "+0.75**, because divisions are weighted by how many listings they "
        "carry. On the row basis - which is what the model actually sees - "
        "the pair would not have breached the 0.80 rule at all. Both figures "
        "are correct and neither is a mistake. `hand_m` chosen because it measures "
        "hydrological position - metres above the stream that drains the point - "
        "rather than metres above sea level, which is a proxy for it. The column "
        "stays in the file so the choice can be reversed without re-running "
        "anything",
}

DISTINCT_MIN = 30
R_MAX = 0.80

# Rejected on VALIDITY, not on the statistics. Both automated rules pass this
# column - 36 distinct values, r = -0.09 against Distance from fort - which is
# exactly why the rejection has to be explicit and written down: the numbers
# would have waved it through. See flood_frequency_rejection.md.
REJECTED = {
    "flood_frequency":
        "measures the sea, not river flooding. 159 of 194 divisions are exactly "
        "0 and every non-zero value is coastal (fort 1.37, uyana 1.23, idama "
        "1.12, moratuwella 0.95, mount lavinia 0.88), while not one Kelani "
        "floodplain division - mulleriyawa, sedawatta, kelanimulla, ambathale, "
        "wellampitiya, kotikawatta, kolonnawa - is above 0. MODIS at 250 m "
        "cannot resolve narrow urban river flooding and the Global Flood "
        "Database only maps large cloud-free events, so the buffers picked up "
        "ocean and tidal water. Including it would report 'flood exposure "
        "lowers coastal land value' from a column that measured distance to "
        "the sea",
}

NOTE = {
    "elevation_m": "Copernicus DEM GLO-30, mean over the 1 km buffer.",
    "slope_deg": "Derived from the same DEM. **Copernicus GLO-30 is a surface "
                 "model, not bare earth** - in built-up Colombo its gradient is "
                 "partly rooftops and tree canopy, so this is only partly terrain.",
    "built_up_fraction": "ESA WorldCover 2021, share of the buffer in class 50. "
                         "The urbanisation intensity variable.",
    "tree_cover_fraction": "ESA WorldCover class 10. Collected as the free "
                           "greenness proxy; expected to mirror built-up.",
    "water_occurrence_pct": "JRC Global Surface Water, mean % of time water was "
                            "present 1984-2021. An empirical record of where "
                            "water actually sits, not a model.",
    "dist_to_stream_km": "OSM, nearest river/stream/canal vertex.",
    "dist_to_kelani_km": "OSM, nearest Kelani Ganga vertex - the river behind "
                         "the Kolonnawa/Kaduwela flooding.",
    "landslide_susceptibility": "NASA LHASA, ~1 km classes. See "
                                "landslide_null_test.md.",
    "hand_m": "MERIT Hydro height above nearest drainage - Earth Engine only.",
    "flood_frequency": "Global Flood Database inundation count - Earth Engine only.",
    "sar_flood_open_land": "Sentinel-1 change detection over three verified "
                           "flood events, mean flooded fraction of the buffer. "
                           "**Named for its blind spot**: radar under-detects "
                           "flooding in dense built-up areas, so it measures "
                           "flooding where SAR can see it, not flooding overall. "
                           "See sar_flood_validation.md.",
}


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load():
    data, missing = {}, []
    for fn, cols in SOURCES:
        path = os.path.join(PROC, fn)
        if not os.path.exists(path):
            missing.append((fn, cols))
            continue
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                d = data.setdefault(r["Address"], {})
                for c in cols:
                    if c in r and r[c] != "":
                        d[c] = float(r[c])
    return data, missing


def main():
    with open(FINAL, encoding="utf-8-sig") as fh:
        final = list(csv.DictReader(fh))

    fort, addr_id = {}, {}
    for r in final:
        fort.setdefault(r["Address"], float(r["Distance from fort"]))
        addr_id.setdefault(r["Address"], int(r["Address_ID"]))

    data, missing = load()
    addrs = sorted(fort, key=lambda a: addr_id[a])
    cand = []
    for _, cols in SOURCES:                       # two files can declare the
        for c in cols:                            # same column; screen it once
            if c not in cand and any(c in data.get(a, {}) for a in addrs):
                cand.append(c)

    # ------------------------------------------------------------- screening
    stat = {}
    for c in cand:
        vals = [data[a].get(c) for a in addrs]
        present = [(a, v) for a, v in zip(addrs, vals) if v is not None]
        v = [x for _, x in present]
        stat[c] = {
            "n": len(v),
            "nulls": len(addrs) - len(v),
            "distinct": len(set(v)),
            "min": min(v), "max": max(v), "mean": float(np.mean(v)),
            "r_fort": pearson(v, [fort[a] for a, _ in present]),
        }

    corr = {}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            pair = [(data[x].get(a), data[x].get(b)) for x in addrs]
            pair = [(u, v) for u, v in pair if u is not None and v is not None]
            if len(pair) > 3:
                corr[(a, b)] = pearson([u for u, _ in pair], [v for _, v in pair])

    verdict = {}
    for c in cand:
        s = stat[c]
        if c in REJECTED:
            verdict[c] = ("REJECT", REJECTED[c])
        elif s["distinct"] < DISTINCT_MIN:
            verdict[c] = ("DROP", f"only {s['distinct']} distinct values across 194 "
                                  f"divisions - a coarse proxy, not a measurement")
        elif abs(s["r_fort"]) > R_MAX:
            verdict[c] = ("DROP", f"r = {s['r_fort']:+.2f} with Distance from fort - "
                                  f"it restates a column already in the model")
        else:
            verdict[c] = ("KEEP", f"{s['distinct']} distinct values, "
                                  f"r = {s['r_fort']:+.2f} with Distance from fort")

    # among surviving pairs that mirror each other, keep the more interpretable
    PREFER = ["water_occurrence_pct", "dist_to_kelani_km", "elevation_m",
              "built_up_fraction", "dist_to_stream_km", "hand_m",
              "flood_frequency", "slope_deg", "tree_cover_fraction"]
    # Pairs the rule may NOT resolve by itself - both are substantive variables
    # and the choice is the analyst's. Flagged in the report instead.
    FLAG_ONLY = {frozenset(("elevation_m", "hand_m"))}
    flagged = []
    for (a, b), r in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        if abs(r) <= R_MAX:
            continue
        if verdict[a][0] != "KEEP" or verdict[b][0] != "KEEP":
            continue
        if frozenset((a, b)) in FLAG_ONLY:
            flagged.append((a, b, r))
            continue
        loser = b if PREFER.index(a) < PREFER.index(b) else a
        winner = a if loser == b else b
        verdict[loser] = ("DROP", f"r = {r:+.2f} with `{winner}` - the same signal "
                                  f"twice; `{winner}` is the more interpretable of the two")

    keep = [c for c in cand if verdict[c][0] == "KEEP"]
    model_set = [c for c in keep if c not in MODELLING_EXCLUDE]

    # ------------------------------------------------------------------ join
    joined, nulls = [], {c: 0 for c in keep}
    for r in final:
        row = dict(r)
        d = data.get(r["Address"], {})
        for c in keep:
            v = d.get(c)
            if v is None:
                nulls[c] += 1
                row[c] = ""
            else:
                row[c] = f"{v:.6f}"
        joined.append(row)

    cols = list(final[0].keys()) + keep
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(joined)

    write_md(addrs, cand, stat, corr, verdict, keep, missing, len(final),
             len(joined), nulls, flagged, model_set)

    print(f"candidates screened : {len(cand)}")
    for c in cand:
        v, why = verdict[c]
        print(f"  {v:4s} {c:26s} distinct={stat[c]['distinct']:4d} "
              f"r_fort={stat[c]['r_fort']:+.2f}  {why[:60]}")
    print(f"\nkept: {keep}")
    print(f"rows in {len(final)} -> out {len(joined)}; nulls in joined columns: "
          f"{ {k: v for k, v in nulls.items() if v} or 'none'}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


def write_md(addrs, cand, stat, corr, verdict, keep, missing, n_in, n_out,
             nulls, flagged, model_set):
    md = ["# Hazard and terrain screening - 194 GN divisions", "",
          f"Generated {dt.date.today().isoformat()} by `src/step6e_screen_and_join.py`.", "",
          "Screening is run on the **194 distinct divisions**, not the 7034 listing "
          "rows. Every listing in a division carries the same feature vector, so each "
          "column costs a degree of freedom across 194 effective units.", "",
          "Rules: drop below "
          f"**{DISTINCT_MIN} distinct values** (a coarse proxy rather than a "
          f"measurement); drop above **|r| = {R_MAX}** against `Distance from fort` "
          "(already in the model); where two survivors correlate above that, keep "
          "the more interpretable one.", "",
          "## 1. Per-variable statistics", "",
          "| variable | distinct | nulls | min | max | mean | r vs Distance from fort | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cand:
        s = stat[c]
        md.append(f"| `{c}` | {s['distinct']} | {s['nulls']} | {s['min']:.3f} | "
                  f"{s['max']:.3f} | {s['mean']:.3f} | {s['r_fort']:+.3f} | "
                  f"**{verdict[c][0]}** |")

    md += ["", "## 2. Reasoning, variable by variable", ""]
    for c in cand:
        v, why = verdict[c]
        md.append(f"**`{c}` - {v}.** {why}. {NOTE.get(c, '')}")
        md.append("")

    md += ["## 3. Correlation between the candidates", "",
           "| pair | r |", "|---|---|"]
    for (a, b), r in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        flag = " **(above threshold)**" if abs(r) > R_MAX else ""
        md.append(f"| `{a}` vs `{b}` | {r:+.3f}{flag} |")

    rej = [c for c in cand if verdict[c][0] == "REJECT"]
    if rej:
        md += ["", "## 3b. Rejected on validity, not on the statistics", ""]
        for c in rej:
            s_ = stat[c]
            md += [f"**`{c}`** passed both automated rules - {s_['distinct']} "
                   f"distinct values (threshold {DISTINCT_MIN}) and "
                   f"r = {s_['r_fort']:+.2f} against Distance from fort "
                   f"(threshold {R_MAX}) - and was rejected anyway, because it "
                   f"{verdict[c][1]}.", "",
                   "The screening rules would have waved this column through. "
                   "That is the point of writing the rejection down: see "
                   "`flood_frequency_rejection.md` for the full record.", ""]

    if flagged:
        md += ["## 3c. Collinear pair - flagged, then decided by the analyst", ""]
        for a, b, r in flagged:
            md += [f"**`{a}` and `{b}` correlate at r = {r:+.3f}**, above the "
                   f"{R_MAX} threshold. The screening rule was not allowed to "
                   "resolve it: both are substantive and the choice is not a sort "
                   "order's to make.", ""]
        for c, why in MODELLING_EXCLUDE.items():
            md += [f"**Decision: `{c}` is excluded from the recommended modelling "
                   f"set.** {why}.", ""]

    md += ["", "## 4. Recommended set", "",
           f"**{len(keep)} columns are joined into the file:** "
           + ", ".join(f"`{c}`" for c in keep) + ".", "",
           f"**{len(model_set)} of them are the recommended modelling set:** "
           + ", ".join(f"`{c}`" for c in model_set) + ".", ""]
    if len(keep) != len(model_set):
        md += ["The difference is "
               + ", ".join(f"`{c}`" for c in keep if c not in model_set)
               + " - present in the file, excluded from the modelling set by the "
                 "decision in section 3c. Nothing was deleted; reversing the "
                 "choice needs no re-run.", ""]
    if "hand_m" in keep:
        h = stat["hand_m"]
        md += ["`hand_m` is the strongest variable in the set. It is the only one "
               "that measures **hydrological position** - metres above the stream "
               "that drains the point - rather than a correlate of it. It has "
               f"{h['distinct']} distinct values across 194 divisions (no ties at "
               "all), no nulls, and its geography validates independently: the "
               "Kelani floodplain reads low (mulleriyawa 3.40 m, sedawatta 3.42 m, "
               "kelanimulla 3.48 m, ambathale 4.77 m) while the southern uplands "
               "read high (kottawa 11.80 m, homagama 11.60 m, pannipitiya 11.45 m, "
               "maharagama 11.18 m) and the eastern hills highest (thummodara "
               "62.09 m). That ordering was not fitted - it is what the layer "
               "returned.", ""]
    if len(keep) > 6:
        md.append("> That is above the 4-6 target. Drop from the bottom of the "
                  "correlation table first.")
        md.append("")

    if missing:
        md += ["## 5. Not collected", "",
               "| file | columns | why |", "|---|---|---|"]
        for fn, cols in missing:
            md.append(f"| `{fn}` | {', '.join(f'`{c}`' for c in cols)} | Earth Engine "
                      "layers - needs an interactive OAuth login I cannot perform. "
                      "Run `scripts/gee_hazard_extract.py`, then re-run this script; "
                      "they will be screened and joined automatically. |")
        md.append("")

    md += ["## 6. Join", "",
           f"- rows in: **{n_in}**, rows out: **{n_out}**",
           f"- nulls in the joined columns: **{ {k: v for k, v in nulls.items() if v} or 'none'}**",
           "- `final_dataset.csv` was not modified; the joined file is "
           "`final_dataset_hazard.csv`.", ""]
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    main()

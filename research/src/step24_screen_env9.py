"""Step 24 - merge day LST / NDVI / NDBI into the dataset, then run the
Appendix F screening rules on all nine environmental and spatial variables.

MERGE.  Key = `Address`, the resolved GN division - the same key step6e used
to join the existing six, and the grouping key of the model. The climate file
is the Earth Engine export (data/processed/gee_climate_vars.csv) when present,
otherwise the Planetary Computer mirror (landsat_climate_194.csv, step23b).
final_dataset_hazard.csv is NOT modified; the merged file is
final_dataset_env9.csv (7,034 rows, all three new columns appended, whether or
not they pass - screening decides what enters the model, not the join).

SCREENING - Appendix F, Table F.1, on the 194 divisions (not the listing
rows). No rule reads the target.

  Rule 1  null count across the 194 divisions. The model has no imputation
          step and the deployed division lookup needs a value for every
          division, so any null fails.
  Rule 2  distinct-value count >= 30 across the 194 divisions.
  Rule 3  (a) |r| <= 0.80 against `Distance from fort`, already in the model
          (b) physical consistency against an independent record - fixed here
              BEFORE any result was seen (PHYSICAL below)
          (c) pairwise |r| > 0.80 between survivors: the variable already in
              the model is kept; between two new ones, PREFER decides.

The six existing variables are screened with the same code so the table covers
all nine, but their Rule 3(b) record is the one already written
(sar_flood_validation.md, water_recurrence_validation.md, hazard_screening.md)
and their retention is not re-decided here: they are in the deployed model.

Outputs
  data/processed/final_dataset_env9.csv
  data/processed/env9_screening.csv     one row per variable
  data/processed/env9_screening.json    passing set, for step25
  data/processed/env9_screening.md      the record, for Appendix F
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
BASE = os.path.join(PROC, "final_dataset_hazard.csv")
GEE = os.path.join(PROC, "gee_climate_vars.csv")
PCM = os.path.join(PROC, "landsat_climate_194.csv")
COG = os.path.join(PROC, "cog_hazard_194.csv")
OUT_CSV = os.path.join(PROC, "final_dataset_env9.csv")
OUT_TAB = os.path.join(PROC, "env9_screening.csv")
OUT_JSON = os.path.join(PROC, "env9_screening.json")
OUT_MD = os.path.join(PROC, "env9_screening.md")

EXISTING = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
            "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]
NEW = ["lst_day_c", "ndvi", "ndbi"]
DISTINCT_MIN = 30
R_MAX = 0.80
PREFER = ["ndvi", "lst_day_c", "ndbi"]   # NDBI last: it confuses bare soil with roofs

# Rule 3(b), written before the extraction finished. Each new variable must
# agree in sign and strength with an independent land-cover product (ESA
# WorldCover 2021, 10 m - a different sensor and a classification, not an
# index) and fall in a physically possible range for lowland Colombo.
PHYSICAL = {
    "ndvi": dict(ref="tree_cover_fraction", sign=+1, r_min=0.50,
                 lo=-0.2, hi=0.9,
                 why="vegetation index must rise with WorldCover tree-cover share"),
    "ndbi": dict(ref="built_up_fraction", sign=+1, r_min=0.50,
                 lo=-0.6, hi=0.4,
                 why="built-up index must rise with WorldCover built-up share"),
    "lst_day_c": dict(ref="built_up_fraction", sign=+1, r_min=0.50,
                      lo=20.0, hi=50.0,
                      why="daytime surface temperature must rise with built-up "
                          "share (urban heat island)"),
}
LABEL = {
    "elevation_m": "Copernicus DEM GLO-30, 1 km buffer mean",
    "water_occurrence_pct": "JRC Global Surface Water occurrence, 1 km buffer mean",
    "dist_to_stream_km": "OSM nearest river/stream/canal",
    "dist_to_kelani_km": "OSM nearest Kelani Ganga vertex",
    "hand_m": "MERIT Hydro height above nearest drainage",
    "sar_flood_open_land": "Sentinel-1 change detection, three verified events",
    "lst_day_c": "Landsat 8/9 C2 L2 ST_B10, median 2023-2025, deg C, 1 km buffer mean",
    "ndvi": "Landsat 8/9 C2 L2 (B5-B4)/(B5+B4), median 2023-2025, 1 km buffer mean",
    "ndbi": "Landsat 8/9 C2 L2 (B6-B5)/(B6+B5), median 2023-2025, 1 km buffer mean",
}


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    base = pd.read_csv(BASE, low_memory=False)
    src = GEE if os.path.exists(GEE) else PCM
    clim = pd.read_csv(src)
    clim["Address"] = clim["Address"].astype(str).str.strip().str.lower()

    # ---------------------------------------------------------------- merge
    assert clim["Address"].is_unique, "climate file has duplicate Address"
    div = set(base["Address"])
    miss = sorted(div - set(clim["Address"]))
    extra = sorted(set(clim["Address"]) - div)
    merged = base.merge(clim[["Address"] + NEW], on="Address", how="left",
                        validate="many_to_one")
    assert len(merged) == len(base)
    merged.to_csv(OUT_CSV, index=False)

    # one row per division - the screening unit
    d = (merged.groupby("Address", sort=False)
         .first()[["Distance from fort"] + EXISTING + NEW])
    cog = pd.read_csv(COG).set_index("Address")[["built_up_fraction",
                                                 "tree_cover_fraction"]]
    d = d.join(cog)
    fort = d["Distance from fort"]

    # ------------------------------------------------------------ rules 1-3a
    rows = []
    for c in EXISTING + NEW:
        v = d[c]
        rows.append(dict(variable=c, block="existing" if c in EXISTING else "new",
                         nulls=int(v.isna().sum()), distinct=int(v.nunique()),
                         min=v.min(), max=v.max(), mean=v.mean(),
                         r_fort=r(v, fort)))
    tab = pd.DataFrame(rows).set_index("variable")
    tab["rule1"] = tab["nulls"] == 0
    tab["rule2"] = tab["distinct"] >= DISTINCT_MIN
    tab["rule3a"] = tab["r_fort"].abs() <= R_MAX

    # ---------------------------------------------------------------- rule 3b
    phys = {}
    for c, p in PHYSICAL.items():
        rr = r(d[c], d[p["ref"]])
        in_range = bool(d[c].min() >= p["lo"] and d[c].max() <= p["hi"])
        ok = bool(np.sign(rr) == p["sign"] and abs(rr) >= p["r_min"] and in_range)
        phys[c] = dict(ref=p["ref"], r=rr, in_range=in_range, ok=ok, why=p["why"],
                       lo=p["lo"], hi=p["hi"], r_min=p["r_min"])
        tab.loc[c, "rule3b"] = ok
    for c in EXISTING:
        tab.loc[c, "rule3b"] = True       # prior validation record, not re-decided
    tab["rule3b"] = tab["rule3b"].astype(bool)

    # ---------------------------------------------------------------- rule 3c
    allv = EXISTING + NEW
    corr = d[allv].corr()
    pairs = []
    for i, a in enumerate(allv):
        for b in allv[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: -abs(t[2]))
    tab["rule3c"] = True
    tab["rule3c_note"] = ""
    survivors = {c for c in allv if tab.loc[c, ["rule1", "rule2", "rule3a", "rule3b"]].all()}
    for a, b, rv in pairs:
        if abs(rv) <= R_MAX or a not in survivors or b not in survivors:
            continue
        if a in EXISTING and b in EXISTING:
            for c, o in ((a, b), (b, a)):
                tab.loc[c, "rule3c_note"] = (f"r = {rv:+.3f} with `{o}`; both already "
                                             "in the model (hazard_screening.md 3c)")
            continue
        if (a in EXISTING) != (b in EXISTING):
            loser, winner = (b, a) if a in EXISTING else (a, b)
            why = "already in the model"
        else:
            loser = b if PREFER.index(a) < PREFER.index(b) else a
            winner = a if loser == b else b
            why = "preferred on interpretability"
        if tab.loc[loser, "rule3c"]:
            tab.loc[loser, "rule3c"] = False
            tab.loc[loser, "rule3c_note"] = (f"r = {rv:+.3f} with `{winner}` "
                                             f"({why}) - the same signal twice")
            survivors.discard(loser)

    tab["passes"] = tab[["rule1", "rule2", "rule3a", "rule3b", "rule3c"]].all(axis=1)
    passing_new = [c for c in NEW if tab.loc[c, "passes"]]
    tab.to_csv(OUT_TAB, float_format="%.6f")

    # ------------------------------------------- GEE vs Planetary Computer
    agree = None
    if os.path.exists(GEE) and os.path.exists(PCM):
        g = pd.read_csv(GEE).assign(Address=lambda x: x.Address.str.strip().str.lower())
        p = pd.read_csv(PCM)
        m = g.merge(p, on="Address", suffixes=("_gee", "_pc"))
        agree = {c: dict(n=len(m), r=r(m[c + "_gee"], m[c + "_pc"]),
                         mad=float((m[c + "_gee"] - m[c + "_pc"]).abs().mean()))
                 for c in NEW}

    info = dict(generated=dt.datetime.now().isoformat(timespec="seconds"),
                climate_source=os.path.basename(src),
                divisions_in_dataset=len(div), climate_rows=len(clim),
                missing_from_climate=miss, extra_in_climate=extra,
                rows_in=len(base), rows_out=len(merged),
                nulls_after_merge={c: int(merged[c].isna().sum()) for c in NEW},
                passing_new=passing_new, env_selected=EXISTING + passing_new,
                physical=phys, gee_vs_pc=agree)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=float)

    write_md(tab, phys, pairs, info, d, clim)
    print(tab[["nulls", "distinct", "r_fort", "rule1", "rule2", "rule3a",
               "rule3b", "rule3c", "passes"]].to_string(float_format="%.3f"))
    for c in NEW:
        print(f"  {c}: physical r vs {phys[c]['ref']} = {phys[c]['r']:+.3f}, "
              f"range ok {phys[c]['in_range']}  note: {tab.loc[c, 'rule3c_note']}")
    print(f"\nsource {src}\npassing new: {passing_new}\nwrote {OUT_CSV}\nwrote {OUT_MD}")


def write_md(tab, phys, pairs, info, d, clim):
    yn = lambda b: "pass" if b else "**FAIL**"
    md = ["# Environmental and spatial variable screening - nine variables, 194 GN divisions", "",
          f"Generated {info['generated'][:10]} by `src/step24_screen_env9.py`. "
          f"Climate source: `{info['climate_source']}`.", "",
          "Appendix F rules, unchanged, run on the 194 divisions. No rule reads the "
          "target. Rule 3(b) for the three new variables was fixed in the script "
          "before the extraction finished.", "",
          "## 1. Merge", "",
          f"- key `Address`; {info['climate_rows']} climate rows against "
          f"{info['divisions_in_dataset']} divisions in the dataset",
          f"- divisions missing from the climate file: {info['missing_from_climate'] or 'none'}",
          f"- rows in {info['rows_in']}, rows out {info['rows_out']}; nulls after merge: "
          f"{info['nulls_after_merge']}", "",
          "## 2. Rules applied to all nine", "",
          "| variable | block | nulls | distinct | min | max | mean | r vs Distance from fort "
          "| R1 nulls | R2 distinct | R3a fort | R3b physical | R3c pairwise | outcome |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in tab.iterrows():
        md.append(f"| `{c}` | {s.block} | {s.nulls} | {s.distinct} | {s['min']:.3f} | "
                  f"{s['max']:.3f} | {s['mean']:.3f} | {s.r_fort:+.3f} | {yn(s.rule1)} | "
                  f"{yn(s.rule2)} | {yn(s.rule3a)} | "
                  f"{'prior record' if s.block == 'existing' else yn(s.rule3b)} | "
                  f"{yn(s.rule3c)} | **{'PASS' if s.passes else 'FAIL'}** |")
    md += ["", "## 3. Rule 3(b) - physical consistency of the new variables", "",
           "| variable | reference (ESA WorldCover 2021) | required | observed r | "
           "plausible range | observed range | result |", "|---|---|---|---|---|---|---|"]
    for c, p in phys.items():
        md.append(f"| `{c}` | `{p['ref']}` | r >= +{p['r_min']:.2f} | {p['r']:+.3f} | "
                  f"{p['lo']} to {p['hi']} | {d[c].min():.3f} to {d[c].max():.3f} | "
                  f"{yn(p['ok'])} |")
    md += ["", "## 4. Pairwise correlations above 0.80", "", "| pair | r |", "|---|---|"]
    for a, b, rv in pairs:
        if abs(rv) > R_MAX:
            md.append(f"| `{a}` vs `{b}` | {rv:+.3f} |")
    md += ["", "Correlations of each new variable with every existing one:", "",
           "| | " + " | ".join(f"`{c}`" for c in NEW) + " |",
           "|---|" + "---|" * len(NEW)]
    cm = d[EXISTING + NEW + ["Distance from fort", "built_up_fraction",
                             "tree_cover_fraction"]].corr()
    for e in EXISTING + NEW + ["Distance from fort", "built_up_fraction",
                               "tree_cover_fraction"]:
        md.append(f"| `{e}` | " + " | ".join(f"{cm.loc[e, c]:+.3f}" for c in NEW) + " |")
    md += ["", "## 5. Reasoning, variable by variable", ""]
    for c, s in tab.iterrows():
        fails = [n for n, k in (("Rule 1 (nulls)", "rule1"), ("Rule 2 (distinct values)", "rule2"),
                                ("Rule 3a (|r| vs Distance from fort > 0.80)", "rule3a"),
                                ("Rule 3b (physical consistency)", "rule3b"),
                                ("Rule 3c (duplicates a retained variable)", "rule3c"))
                 if not s[k]]
        extra = f" {s.rule3c_note}." if s.rule3c_note else ""
        verdict = "PASS" if s.passes else "FAIL - " + "; ".join(fails)
        md.append(f"**`{c}` - {verdict}.** {LABEL[c]}. {s.distinct} distinct values, "
                  f"r = {s.r_fort:+.2f} with Distance from fort.{extra}")
        md.append("")
    if "water_share" in clim:
        w = clim.set_index("Address")["water_share"].sort_values(ascending=False)
        md += ["## 6. Diagnostic - water inside the 1 km buffers", "",
               "Not a candidate. The buffer means are unmasked, exactly like the existing "
               "six, so coastal and lakeside buffers mix water pixels into NDVI, NDBI and "
               "LST (water has negative NDVI and a cool daytime surface). Divisions whose "
               "buffer is more than 10% water:", "",
               ", ".join(f"{a} {v:.2f}" for a, v in w[w > 0.10].items()) or "none", ""]
    if info["gee_vs_pc"]:
        md += ["## 7. Earth Engine export vs Planetary Computer mirror", "",
               "| variable | divisions | r | mean abs difference |", "|---|---|---|---|"]
        for c, a in info["gee_vs_pc"].items():
            md.append(f"| `{c}` | {a['n']} | {a['r']:.4f} | {a['mad']:.4f} |")
        md.append("")
    md += ["## 8. Result", "",
           f"**New variables passing all rules: "
           f"{', '.join(f'`{c}`' for c in info['passing_new']) or 'none'}.**", "",
           f"Environmental block carried to step25: "
           f"{', '.join(f'`{c}`' for c in info['env_selected'])}.", ""]
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    main()

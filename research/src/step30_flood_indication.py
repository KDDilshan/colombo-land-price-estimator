"""Step 30 - GN-level "past flooding" indication, from official records only,
and a sanity check of it against the satellite-derived variables.

WHAT THIS IS.  Descriptive information shown next to the predicted price: has
this GN division been recorded as flooded before, by whom, and when. It is NOT
a prediction, is not used by the price model, and says nothing about cause.
The absence of a record does not mean the land cannot flood.

INPUTS (all Sri Lankan government sources)
  official_flood_gn.csv          step28 - Survey Department flood maps,
                                 May 2016, May 2018 and Cyclone Ditwah Nov 2025:
                                 share of each GN polygon inside the extent
  dmc_flood_records_colombo.csv  step29 - DMC DesInventar, 1974-2019. GN names
                                 are parsed from the free-text place field
                                 ("Sedawatta(GN)") and matched, within the
                                 record's DS division, to the ADM4 member names
                                 of each GN polygon.

INDICATION - rule fixed here, before the table was looked at
  events  = [2016 share >= 1%] + [2018 share >= 1%] + [2025 share >= 1%]
            + distinct DMC years naming the GN
  share   = max(2016, 2018, 2025 share)
  High    share >= 25%  or  events >= 3
  Medium  any official record (events >= 1)
  Low     inside the coverage of at least one Survey Department map, not flooded
          in it, and not named in any DMC record
  No official record   none of the above (outside both maps, never named)

SANITY CHECK (task 6)
On the 194 divisions with satellite variables: does the official record agree
with sar_flood_open_land, water_occurrence_pct and hand_m? Reported as rank
correlations and group medians. Nothing is tuned to make them agree.

Outputs
  data/processed/flood_indication_gn.csv
  data/processed/flood_indication_report.md
  model test/deploy/flood_records.csv       read by the app
"""

import difflib
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
DEPLOY = os.path.join(ROOT, "model test", "deploy")
DS_ALIAS = {"hanwella": "seethawaka"}          # DMC's name for Seethawaka DS
# DMC place names whose nearest GN name is a different, real GN - a match
# would be a guess, so they stay unmatched (and are listed in the report)
REJECT = {"medapitiya",        # nearest: Madampitiya
          "samanthapura"}      # nearest: Sammanthranapura
PHYS = ["hand_m", "sar_flood_open_land", "water_occurrence_pct", "dist_to_kelani_km",
        "dist_to_stream_km", "elevation_m"]


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def dmc_gn_matches(gn):
    rec = pd.read_csv(os.path.join(PROC, "dmc_flood_records_colombo.csv"))
    rec = rec[(rec.event_class == "flood") & rec.lugar.notna()]
    by_ds = {}
    for _, g in gn.iterrows():
        for ds in str(g.ds_divisions).split("; "):
            for m in str(g.adm4_members).split("; ") + [g.address]:
                by_ds.setdefault(norm(ds), {}).setdefault(norm(m), set()).add(g.address)
    district = {}
    for d in by_ds.values():
        for k, v in d.items():
            district.setdefault(k, set()).update(v)

    def lookup(nm, names, suffix=""):
        """Exact/fuzzy within the DS; then 'X' -> 'X East'/'X West' prefix
        members; then the same two tests district-wide (DMC and ADM4 do not
        always agree on which DS a GN sits in). When the DMC text ends in a
        separate suffix word ('Maha Buthgamuwa D', 'Malabe North'), the match
        must end in the same suffix: a different letter is a different GN."""
        if nm in REJECT:
            return None, set()
        for pool in (names, district):
            best = difflib.get_close_matches(nm, list(pool), n=1, cutoff=0.85)
            if best and (not suffix or best[0].endswith(suffix)):
                return best[0], pool[best[0]]
            pre = [k for k in pool if len(nm) >= 6 and k.startswith(nm)
                   and re.fullmatch(r"(north|south|east|west|[a-e]|\d)+", k[len(nm):])]
            if pre:
                return "|".join(pre), set().union(*(pool[k] for k in pre))
        return None, set()

    hits, unmatched = [], []
    for _, r in rec.iterrows():
        ds = norm(DS_ALIAS.get(norm(r.ds_division), r.ds_division))
        names = by_ds.get(ds, {})
        text = str(r.lugar)
        if not re.search(r"g\.?\s*n", text, re.I):
            continue                         # only places the DMC tagged as a GN
        # "A(GN) B(GN)", "A (GN), B(GN" - the tag itself is the separator
        for part in re.split(r"\(\s*g\.?\s*n\.?\s*\)?|[,/&;]| and ", text, flags=re.I):
            nm = re.sub(r"\d+|watta$", "", norm(part)) if norm(part).endswith("watta") and \
                re.search(r"\d", part) else re.sub(r"\d+", "", norm(part))
            if len(nm) < 4:
                continue
            m = re.search(r"\s(north|south|east|west|[a-e])\s*$", part.strip(), re.I)
            key, addrs = lookup(nm, names, m.group(1).lower() if m else "")
            if addrs:
                for a in addrs:
                    hits.append(dict(address=a, year=int(r.fechano), dmc_place=part.strip(),
                                     matched_name=key, serial=r.serial))
            else:
                unmatched.append((r.ds_division, part.strip()))
    return pd.DataFrame(hits), unmatched, len(rec)


def main():
    gn = pd.read_csv(os.path.join(PROC, "official_flood_gn.csv"))
    hits, unmatched, n_rec = dmc_gn_matches(gn)
    yrs = (hits.groupby("address").year.apply(lambda s: sorted(set(s))) if len(hits)
           else pd.Series(dtype=object))
    gn["dmc_years"] = gn.address.map(yrs).apply(lambda v: v if isinstance(v, list) else [])
    gn["dmc_n_years"] = gn.dmc_years.apply(len)

    ds = pd.read_csv(os.path.join(PROC, "dmc_flood_by_ds.csv")).set_index("ds_division")
    ds.index = [norm(DS_ALIAS.get(norm(i), i)) for i in ds.index]
    gn["dmc_ds_flood_records"] = gn.ds_divisions.apply(
        lambda s: int(sum(ds.flood_records.get(norm(d), 0) for d in str(s).split("; "))))

    maps = ("2016", "2018", "2025")
    gn["events"] = sum((gn[f"flooded_share_{y}"] >= 0.01).astype(int) for y in maps) + gn.dmc_n_years
    gn["max_share"] = gn[[f"flooded_share_{y}" for y in maps]].max(axis=1)
    covered = np.logical_or.reduce([gn[f"covered_{y}"] > 0.5 for y in maps])
    gn["indication"] = np.select(
        [(gn.max_share >= 0.25) | (gn.events >= 3), gn.events >= 1, covered],
        ["High", "Medium", "Low"], default="No official record")

    # --------------------------------------------------------- sanity check
    base = pd.read_csv(os.path.join(PROC, "final_dataset_env9.csv"), low_memory=False)
    phys = base.groupby("Address")[PHYS].first()
    j = gn.set_index("address").join(phys, how="inner")
    j["flooded_any"] = j.events >= 1
    jc = j[j.indication != "No official record"]
    check = []
    for c in PHYS:
        rho, p = spearmanr(jc.max_share, jc[c])
        a, b = jc[jc.flooded_any][c], jc[~jc.flooded_any][c]
        u = mannwhitneyu(a, b) if len(a) and len(b) else None
        check.append(dict(variable=c, spearman_vs_max_share=rho, p=p,
                          median_flooded=a.median(), median_not_flooded=b.median(),
                          n_flooded=len(a), n_not=len(b),
                          mw_p=u.pvalue if u else np.nan))
    check = pd.DataFrame(check)

    out = gn[["address", "modelled", "ds_divisions", "covered_2016", "flooded_share_2016",
              "covered_2018", "flooded_share_2018", "covered_2025", "flooded_share_2025",
              "dmc_years", "dmc_n_years",
              "dmc_ds_flood_records", "events", "max_share", "indication"]].copy()
    out["dmc_years"] = out.dmc_years.apply(lambda v: " ".join(map(str, v)))
    out.to_csv(os.path.join(PROC, "flood_indication_gn.csv"), index=False)
    out.to_csv(os.path.join(DEPLOY, "flood_records.csv"), index=False)
    if len(hits):
        hits.to_csv(os.path.join(PROC, "dmc_gn_matches.csv"), index=False)

    vc = out.indication.value_counts()
    vm = out[out.modelled].indication.value_counts()
    md = ["# Past flooding by GN division - official records", "",
          "Descriptive information for the app, not a prediction. Sources: Survey "
          "Department of Sri Lanka flood maps (May 2016; May 2018; Cyclone Ditwah, Nov 2025) and the "
          "DMC DesInventar database (1974-2019).", "",
          "## Rule", "",
          "events = [2016 share ≥ 1%] + [2018 share ≥ 1%] + [2025 share ≥ 1%] + distinct DMC years naming the GN; "
          "High if max share ≥ 25% or events ≥ 3; Medium if any record; Low if inside a "
          "Survey Department map's coverage and never recorded; otherwise No official record.", "",
          "## Result", "",
          "| indication | all 443 GN units | 187 modelled |", "|---|---|---|"]
    for k in ("High", "Medium", "Low", "No official record"):
        md.append(f"| {k} | {vc.get(k, 0)} | {vm.get(k, 0)} |")
    md += ["", f"DMC: {n_rec} Colombo flood records with a place field; "
           f"{len(hits)} GN-name matches to {hits.address.nunique() if len(hits) else 0} GN units; "
           f"{len(unmatched)} GN-tagged names not matched (listed in section 4).", "",
           "## Sanity check against the satellite variables (task 6)", "",
           f"{len(jc)} divisions with satellite variables and official coverage. "
           "Official flooded share against each variable:", "",
           "| variable | Spearman ρ vs max official share | median, recorded flooded | "
           "median, not recorded | Mann-Whitney p |", "|---|---|---|---|---|"]
    for _, c in check.iterrows():
        md.append(f"| `{c.variable}` | {c.spearman_vs_max_share:+.3f} (p {c.p:.1e}) | "
                  f"{c.median_flooded:.3f} (n {c.n_flooded}) | {c.median_not_flooded:.3f} "
                  f"(n {c.n_not}) | {c.mw_p:.1e} |")
    md += ["", "## 4. DMC GN names not matched", "",
           ", ".join(f"{d}: {n}" for d, n in unmatched[:80]) or "none", ""]
    open(os.path.join(PROC, "flood_indication_report.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()

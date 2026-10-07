"""Step 1a - Build the Colombo district GN division list and its coordinates.

Source: OCHA/HDX Common Operational Dataset `cod-ab-lka` (downloaded by
`step0_fetch_gn_boundaries.py`).  ADM4 = Grama Niladhari division,
ADM3 = Divisional Secretariat division, ADM2 = district.

What this does
--------------
1. Selects every ADM4 feature whose ADM2 is Colombo -> 557 source divisions
   across 13 DS divisions, matching section 3.2.1 of the dissertation.

2. Reduces each source name to its *base name* by repeatedly stripping the
   qualifiers that distinguish sub-divisions of one place from each other.  The
   dissertation motivates this by the fact that an ikman.lk advertiser writes
   the base name only ("Kotahena", never "Kotahena East"), so the sub-division
   is unrecoverable from the advertisement.  Four qualifier families occur in
   the Colombo ADM4 names:

     cardinal suffix    North / South / East / West / Central
     letter suffix      A / B / C, sometimes quoted ('A')
     town suffix        Town / New Town
     positional prefix  Ihala, Pahala, Maha, Kuda, Megoda, Meda, Udu
                        (upper / lower / greater / lesser / far-side / middle)

   Stripping is iterative because names stack qualifiers, e.g.
   "Ihala Kosgama North" -> "Ihala Kosgama" -> "Kosgama".

   Merging happens *within a DS division*, as the dissertation states the
   exercise was carried out per DS division.  This yields 399 divisions -
   the exact figure reported in section 3.2.1.

   See NOTES.md for the evidence that each of the four families (not only the
   cardinal one) is part of the original author's merge.

3. Gives every merged division one coordinate: the area-weighted mean of the
   ADM4 polygon centroids merged into it.  A division with a single source
   division therefore keeps its own polygon centroid unchanged.

Outputs
-------
data/reference/gn_divisions_raw_557.csv     one row per source ADM4 division
data/reference/gn_divisions_merged_399.csv  one row per merged division
data/reference/gn_divisions_by_ds_summary.csv
data/reference/gn_lookup.csv                name -> single coordinate, the
                                            table every later step joins on

Run:  python src/step1a_build_gn_divisions.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADM4 = ROOT / "data" / "raw" / "lka_admin_boundaries.geojson" / "lka_admin4.geojson"
OUT = ROOT / "data" / "reference"

DISTRICT = "Colombo"

CARDINAL_RE = re.compile(
    r"^(?P<base>.+?)[\s\-]+(?P<q>north|south|east|west|central)$", re.IGNORECASE
)
LETTER_RE = re.compile(r"^(?P<base>.+?)[\s\-]+'?(?P<q>[ABC])'?$")
TOWN_RE = re.compile(r"^(?P<base>.+?)\s+(?P<q>(?:new\s+)?town)$", re.IGNORECASE)
PREFIX_RE = re.compile(
    r"^(?P<q>ihala|pahala|maha|kuda|megoda|meda|udu)\s+(?P<base>.+)$", re.IGNORECASE
)

STRIPPERS = (
    ("letter", LETTER_RE),
    ("town", TOWN_RE),
    ("cardinal", CARDINAL_RE),
    ("prefix", PREFIX_RE),
)

# Names that end in a qualifier word but are NOT a qualified form of a shorter
# name - there is no "Havelock" GN division for "Havelock Town" to merge into,
# and the original author's own address list carries "havelock town" verbatim.
# Protecting it does not change the division count, only the label.
PROTECTED_NAMES = {"havelock town"}


def reduce_name(name: str) -> tuple[str, list[str]]:
    """Reduce a GN division name to its base name.

    Returns (base_name, qualifiers_stripped_in_order).
    """
    s = " ".join(str(name).split())
    if s.lower() in PROTECTED_NAMES:
        return s, []
    stripped: list[str] = []
    changed = True
    while changed:
        changed = False
        for _, rx in STRIPPERS:
            m = rx.match(s)
            if m:
                base = m.group("base").strip(" -")
                if not base:  # the qualifier IS the whole name - keep as-is
                    continue
                stripped.append(m.group("q"))
                s = base
                changed = True
                break
    return s, stripped


def load_colombo_adm4() -> pd.DataFrame:
    if not ADM4.exists():
        raise SystemExit(
            f"Missing {ADM4}.\nRun: python src/step0_fetch_gn_boundaries.py first."
        )
    with open(ADM4, encoding="utf-8") as fh:
        gj = json.load(fh)

    rows = []
    for feat in gj["features"]:
        p = feat["properties"]
        if p.get("adm2_name") != DISTRICT:
            continue
        rows.append(
            {
                "gn_pcode": p["adm4_pcode"],
                "gn_name_source": p["adm4_name"],
                "gn_name_si": p.get("adm4_name1"),
                "gn_name_ta": p.get("adm4_name2"),
                "ds_name": p["adm3_name"],
                "ds_pcode": p["adm3_pcode"],
                "district": p["adm2_name"],
                "area_sqkm": p.get("area_sqkm"),
                "lat": p.get("center_lat"),
                "lon": p.get("center_lon"),
            }
        )
    return pd.DataFrame(rows).sort_values(["ds_pcode", "gn_pcode"]).reset_index(drop=True)


def weighted_centroid(g: pd.DataFrame) -> tuple[float, float]:
    w = g["area_sqkm"].fillna(0.0)
    if w.sum() > 0:
        return (
            float((g["lat"] * w).sum() / w.sum()),
            float((g["lon"] * w).sum() / w.sum()),
        )
    return float(g["lat"].mean()), float(g["lon"].mean())


def main() -> int:
    raw = load_colombo_adm4()
    print(f"ADM4 features in {DISTRICT} district         : {len(raw)}")
    print(f"DS divisions (ADM3)                       : {raw['ds_name'].nunique()}")

    bad = raw[raw["lat"].isna() | raw["lon"].isna()]
    if len(bad):
        print(f"WARNING: {len(bad)} ADM4 rows lack a centroid")
        print(bad[["gn_pcode", "gn_name_source", "ds_name"]].to_string(index=False))

    reduced = raw["gn_name_source"].map(reduce_name)
    raw["base_name"] = [b for b, _ in reduced]
    raw["qualifiers_stripped"] = [" > ".join(q) for _, q in reduced]

    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "gn_divisions_raw_557.csv", index=False, encoding="utf-8")

    merged_rows = []
    for (ds_pcode, ds_name, base), g in raw.groupby(
        ["ds_pcode", "ds_name", "base_name"], sort=False
    ):
        lat, lon = weighted_centroid(g)
        merged_rows.append(
            {
                "gn_division": base,
                "ds_division": ds_name,
                "ds_pcode": ds_pcode,
                "n_source_divisions": len(g),
                "merged_from": " | ".join(g["gn_name_source"].tolist()),
                "gn_names_si": " | ".join(sorted(set(g["gn_name_si"].dropna()))),
                "area_sqkm": float(g["area_sqkm"].fillna(0.0).sum()),
                "lat": lat,
                "lon": lon,
            }
        )

    merged = (
        pd.DataFrame(merged_rows)
        .sort_values(["ds_division", "gn_division"])
        .reset_index(drop=True)
    )
    merged.insert(0, "gn_id", range(1, len(merged) + 1))
    merged.to_csv(OUT / "gn_divisions_merged_399.csv", index=False, encoding="utf-8")

    print(f"Merged distinct GN divisions (per DS)     : {len(merged)}")
    print(f"  formed by merging more than one ADM4    : {(merged.n_source_divisions > 1).sum()}")

    summary = (
        merged.groupby("ds_division")
        .agg(
            gn_divisions_source=("n_source_divisions", "sum"),
            gn_divisions_merged=("gn_division", "count"),
        )
        .sort_index()
    )
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(OUT / "gn_divisions_by_ds_summary.csv", encoding="utf-8")
    print("\nRevised number of GN divisions by DS division (cf. Table 3.6):")
    print(summary.to_string())

    # ---- name-level lookup -------------------------------------------------
    # A web advertisement carries a place name and no DS division, so the name
    # is the only key available at mapping time.  Base names shared by two DS
    # divisions are therefore collapsed once more into a single coordinate.
    lookup_rows = []
    for base, g in merged.groupby("gn_division", sort=True):
        lat, lon = weighted_centroid(g)
        lookup_rows.append(
            {
                "gn_division": base,
                "n_ds_divisions": len(g),
                "ds_divisions": " | ".join(sorted(g["ds_division"])),
                "n_source_divisions": int(g["n_source_divisions"].sum()),
                "area_sqkm": float(g["area_sqkm"].sum()),
                "lat": lat,
                "lon": lon,
            }
        )
    lookup = pd.DataFrame(lookup_rows).reset_index(drop=True)
    lookup.to_csv(OUT / "gn_lookup.csv", index=False, encoding="utf-8")

    collisions = lookup[lookup.n_ds_divisions > 1]
    print(f"\nDistinct GN base names (mapping keys)     : {len(lookup)}")
    print(f"  names spanning >1 DS division, collapsed: {len(collisions)}")
    if len(collisions):
        print(collisions[["gn_division", "ds_divisions", "n_source_divisions"]].to_string(index=False))

    for f in (
        "gn_divisions_raw_557.csv",
        "gn_divisions_merged_399.csv",
        "gn_divisions_by_ds_summary.csv",
        "gn_lookup.csv",
    ):
        print(f"Wrote {OUT / f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

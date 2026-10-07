"""Step 1e - Apply the agreed cleaning rules from the step 1d price audit.

Decisions taken (see NOTES.md section 3.7):

  R1  APPLY  - drop rent / lease listings
  R2  APPLY  - drop non-land records (apartments, offices, warehouses)
  R3  APPLY  - drop listings whose Title names a town in another district
  R4  NOT APPLIED - the 16 rows at the Rs 10,000,000 per-perch ceiling are
      KEPT and marked with a new boolean column `price_censored`.  The target
      is right-censored and that has to be stated, not deleted.
  R5  DROPPED, NOT REPAIRED - re-tagging a row's price scale because the
      figure sits near the division median is circular: it would assign the
      price from the location, and the model then predicts price from
      location.  These rows are removed and counted.
  R7  applied upstream in step1b (Title-derived out-of-district match ranks
      above a postal match taken from the Town tag).

Order matters.  R1-R3 run first, because rentals and apartments distort the
per-division medians that R5 is measured against.  R5 medians are therefore
computed on the R1-R3 survivors.

Outputs
-------
data/interim/listings_gn_clean.csv     the cleaned listing table
results/step1e_dropped_rows.csv        every dropped row + the rule that hit it
results/step1e_cleaning_report.txt     counts at each stage
results/step1e_division_summary.csv    n, median, IQR of Price_per_Perch per GN
results/step1e_division_summary.txt    same, human readable

Run:  python src/step1e_clean_listings.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"

CENSOR_CEILING = 10_000_000       # Price_per_Perch is clipped here upstream
CENSOR_TOLERANCE = 9_999_999      # treat anything at/above this as censored

RATIO_LIMIT = 10.0                # R5: "more than 10x below the division median"
SCALE_SWAP_LO, SCALE_SWAP_HI = 1 / 3, 3.0

RENT_WORDS = [
    "rent", "renting", "for rent", "lease", "leasing", "leasehold",
    "monthly", "per month", "rental", "p/m",
]
NON_LAND_TITLE_WORDS = [
    "apartment", "flat", "annex", "annexe", "condominium", "condo",
    "penthouse", "storey house", "storied house", "bedroom", "br house",
    "villa", "bungalow", "mansion", "hotel", "guest house", "warehouse",
    "factory", "showroom", "shop", "office space", "building for",
]
NON_LAND_TYPES = [
    "Office", "Building", "Factory / Workshop", "Warehouse / Storage",
    "Hotel", "Shop",
]

# R11 - a re-posted advert repeats every one of these.  Note that all of them
# except Title and Address_Raw are columns the model itself sees, so duplicate
# rows are identical in feature space and in the target.
DEDUP_KEY = [
    "Title", "Address_Raw", "Land_Size_Perch", "Price_LKR",
    "Land_Type", "gn_division",
]


def _has(text, words) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(w in t for w in words)


def load_other_district_places() -> dict[str, str]:
    places: dict[str, str] = {}
    man = REF / "out_of_district_manual.csv"
    if man.exists():
        for r in pd.read_csv(man).itertuples():
            if isinstance(r.place, str):
                places[r.place.strip().lower()] = str(r.district)
    geo = REF / "geocode_out_of_district.csv"
    if geo.exists():
        for r in pd.read_csv(geo).itertuples():
            if isinstance(r.alias, str) and len(r.alias.split()) <= 2:
                places[r.alias.strip().lower()] = str(getattr(r, "osm_district", ""))
    places.setdefault("polgahawella", "Kurunegala")
    return places


OTHER_DISTRICT_PLACES = load_other_district_places()


def title_names_other_district(title):
    if not isinstance(title, str):
        return None
    toks = set(re.sub(r"[^\w]+", " ", title.lower()).split())
    for place, district in OTHER_DISTRICT_PLACES.items():
        if place in toks:
            return f"{place.title()} ({district})"
    return None


def iqr_block(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return len(s), s.median(), q1, q3, q3 - q1


def main() -> int:
    src = INTERIM / "listings_gn_mapped.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run step1b_map_listings_to_gn.py first.")

    merged = pd.read_csv(REF / "gn_divisions_merged_399.csv")
    lookup = pd.read_csv(REF / "gn_lookup.csv")
    n_div_ds = len(merged)          # 399, counted within DS division
    n_div_names = len(lookup)       # 390 distinct names
    n_multi = int((lookup.n_ds_divisions > 1).sum())

    raw = pd.read_csv(src, low_memory=False)
    df = raw[raw.gn_division.notna()].copy().reset_index(drop=True)
    df["row_id"] = df.index
    start_n = len(df)

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)

    emit("=" * 78)
    emit("STEP 1e - CLEANING (agreed rules from the step 1d audit)")
    emit("=" * 78)
    emit(f"input : {src.relative_to(ROOT)}")
    emit(f"rows in, mapped to a GN division : {start_n}")
    emit("")
    emit("GN DIVISION COUNTS - both figures, stated consistently:")
    emit(f"  {n_div_ds} GN divisions counted within DS division (the dissertation's figure)")
    emit(f"  {n_div_names} distinct GN division names")
    emit(f"  {n_multi} names span more than one DS division, which accounts for the")
    emit(f"  {n_div_ds - n_div_names}-row difference (Udumulla spans three, the other seven span two).")
    emit("  A listing names a place, not a DS division, so the mapping key is the")
    emit(f"  NAME: {n_div_names} possible values. The {n_div_ds} figure is the division count.")
    emit("")

    # ------------------------------------------------------------------ R1-R3
    dropped: list[pd.DataFrame] = []

    def apply_rule(mask: pd.Series, rule: str, note_col=None) -> None:
        nonlocal df
        hit = df[mask].copy()
        if len(hit):
            hit["drop_rule"] = rule
            hit["drop_note"] = (
                hit[note_col] if note_col is not None else ""
            ) if isinstance(note_col, str) else (
                note_col(hit) if callable(note_col) else ""
            )
            dropped.append(hit)
        df = df[~mask].copy()

    emit("=" * 78)
    emit("REMOVALS")
    emit("=" * 78)

    m = df.Title.map(lambda t: _has(t, RENT_WORDS))
    n1 = int(m.sum())
    apply_rule(m, "R1 rent/lease listing")
    emit(f"R1  rent / lease listing                 : -{n1:>5}   remaining {len(df):>5}")

    m = df.Land_Type.isin(NON_LAND_TYPES) | df.Title.map(
        lambda t: _has(t, NON_LAND_TITLE_WORDS)
    )
    n2 = int(m.sum())
    apply_rule(m, "R2 non-land record")
    emit(f"R2  non-land record (apartment/office)   : -{n2:>5}   remaining {len(df):>5}")

    m = df.Title.map(lambda t: title_names_other_district(t) is not None)
    n3 = int(m.sum())
    apply_rule(
        m, "R3 title names another district",
        lambda h: h.Title.map(title_names_other_district),
    )
    emit(f"R3  Title names another district         : -{n3:>5}   remaining {len(df):>5}")
    emit("")

    # ------------------------------------------------------------------ R5
    emit("=" * 78)
    emit("R5 - SCALE-MISLABELLED ROWS (DROPPED, NOT RE-TAGGED)")
    emit("=" * 78)
    emit("Re-tagging these would set a row's price scale from how close the")
    emit("figure sits to its own division's median - i.e. it would infer price")
    emit("from location, and the model then predicts price from location. That")
    emit("is circular, so the rows are dropped instead.")
    emit("")
    emit("Definition, applied to the R1-R3 survivors:")
    emit("  Price_Scale == 'total'")
    emit(f"  AND implied per-perch < 1/{RATIO_LIMIT:.0f} of the division's 'per perch' median")
    emit(f"  AND Price_LKR read as a per-perch figure lands within "
         f"{SCALE_SWAP_LO:.2f}x-{SCALE_SWAP_HI:.0f}x of that median")
    emit("")

    df["implied_ppp"] = df.Price_LKR / df.Land_Size_Perch
    pp_median = (
        df[df.Price_Scale == "per perch"].groupby("gn_division").Price_per_Perch.median()
    )
    div_med = df.gn_division.map(pp_median)
    ratio = df.implied_ppp / div_med
    swap = df.Price_LKR / div_med

    m = (
        (df.Price_Scale == "total")
        & div_med.notna()
        & (div_med > 0)
        & (ratio < 1 / RATIO_LIMIT)
        & (swap >= SCALE_SWAP_LO)
        & (swap <= SCALE_SWAP_HI)
    )
    n5 = int(m.sum())
    emit(f"rows matching : {n5}")
    emit("")
    if n5:
        emit(f"{'implied/pch':>13}  {'div median':>13}  {'as per-perch':>14}  {'size':>7}  GN division")
        emit("-" * 84)
        for r in df[m].itertuples():
            emit(f"{r.implied_ppp:>13,.0f}  {pp_median[r.gn_division]:>13,.0f}  "
                 f"{r.Price_LKR:>14,.0f}  {r.Land_Size_Perch:>7.1f}  {r.gn_division}")
    apply_rule(m, "R5 scale mislabelled (dropped)")
    emit("")
    emit(f"R5  scale mislabelled                    : -{n5:>5}   remaining {len(df):>5}")
    emit("")

    # ------------------------------------------------------------------ R10
    emit("=" * 78)
    emit("R10 - 'per acre' ROWS DROPPED")
    emit("=" * 78)
    emit("Price_LKR on these rows is an ACRE price, but Price_per_Perch was")
    emit("computed upstream as Price_LKR / Land_Size_Perch, which is neither a")
    emit("per-acre nor a per-perch figure. The advertised scale cannot be")
    emit("recovered from the columns available, so the rows are unusable.")
    emit("")
    m = df.Price_Scale == "per acre"
    n10 = int(m.sum())
    if n10:
        emit(f"{'Price_per_Perch':>16}  {'Price_LKR':>14}  {'size':>8}  GN division")
        emit("-" * 62)
        for r in df[m].sort_values("Price_per_Perch").itertuples():
            emit(f"{r.Price_per_Perch:>16,.0f}  {r.Price_LKR:>14,.0f}  "
                 f"{r.Land_Size_Perch:>8.1f}  {r.gn_division}")
        emit("")
        emit(f"range of the affected Price_per_Perch values: "
             f"Rs {df[m].Price_per_Perch.min():,.0f} to Rs {df[m].Price_per_Perch.max():,.0f}")
    apply_rule(m, "R10 per acre scale (unrecoverable)")
    emit("")
    emit(f"R10 'per acre' rows                      : -{n10:>5}   remaining {len(df):>5}")
    emit("")

    # ------------------------------------------------------------------ R11
    emit("=" * 78)
    emit("R11 - DE-DUPLICATION OF RE-POSTED ADVERTS")
    emit("=" * 78)
    emit("Key: " + " + ".join(DEDUP_KEY))
    emit("Kept: the row with the earliest Posted_Date (rows with no date sort")
    emit("last, so a dated row always wins).")
    emit("")
    emit("This is a DELIBERATE DEVIATION from the original study, which did not")
    emit("de-duplicate. Rationale: under a random 70/30 split an advert present")
    emit("44 times places copies in both train and test, so the model is scored")
    emit("on rows it memorised. Every column in the key is a column the model")
    emit("sees, so duplicate rows are identical in BOTH feature space and")
    emit("target - keeping them adds no information and inflates every metric.")
    emit("See results/step1f_dedup_diagnostic.txt for the evidence that these")
    emit("are re-posts rather than genuinely subdivided developments.")
    emit("")

    # Snapshot the population R11 acts on, so step1f can re-derive the
    # diagnostic without re-running the whole pipeline.
    df.to_csv(INTERIM / "listings_gn_predup.csv", index=False, encoding="utf-8")

    key_frame = df[DEDUP_KEY].astype(object).where(df[DEDUP_KEY].notna(), "<NA>")
    df["_dedup_key"] = key_frame.astype(str).agg("|".join, axis=1)
    df["_date_sort"] = pd.to_datetime(df.Posted_Date, errors="coerce")

    grp_sizes = df.groupby("_dedup_key").size()
    n_groups = int((grp_sizes > 1).sum())
    n_group_rows = int(grp_sizes[grp_sizes > 1].sum())

    ordered = df.sort_values(["_dedup_key", "_date_sort"], na_position="last")
    keep_idx = ordered.groupby("_dedup_key", sort=False).head(1).index
    m = ~df.index.isin(keep_idx)
    m = pd.Series(m, index=df.index)
    n11 = int(m.sum())

    emit(f"distinct key groups                 : {len(grp_sizes)}")
    emit(f"groups holding more than one row    : {n_groups}")
    emit(f"rows inside those groups            : {n_group_rows}")
    emit(f"largest group                       : {int(grp_sizes.max())} rows")
    emit("")
    emit("group size distribution:")
    for size, n in grp_sizes[grp_sizes > 1].value_counts().sort_index().items():
        emit(f"  {size:>4} rows x {n:>4} groups = {size * n:>5} rows")
    emit("")
    apply_rule(m, "R11 re-posted duplicate advert")
    emit(f"R11 duplicate re-posts                   : -{n11:>5}   remaining {len(df):>5}")
    emit("")
    df = df.drop(columns=["_dedup_key", "_date_sort"])

    # ------------------------------------------------------------------ R4
    emit("=" * 78)
    emit("R4 - NOT APPLIED: CENSORED PRICES KEPT AND MARKED")
    emit("=" * 78)
    df["price_censored"] = (df.Price_per_Perch >= CENSOR_TOLERANCE).astype(int)
    n_cens = int(df.price_censored.sum())
    emit(f"Price_per_Perch is clipped upstream at Rs {CENSOR_CEILING:,}.")
    emit(f"Rows retained and flagged price_censored = 1 : {n_cens}")
    emit("Their true price is unknown and at least this value; the target")
    emit("variable is therefore RIGHT-CENSORED. See NOTES.md section 3.8.")
    if n_cens:
        emit("")
        emit("Censored rows by GN division:")
        for gn, n in df[df.price_censored == 1].gn_division.value_counts().items():
            emit(f"  {n:>4}  {gn}")
    emit("")

    # ------------------------------------------------------------------ save
    df = df.drop(columns=["implied_ppp"]).reset_index(drop=True)
    df["row_id"] = df.index
    df.to_csv(INTERIM / "listings_gn_clean.csv", index=False, encoding="utf-8")

    drop_df = (
        pd.concat(dropped, ignore_index=True)
        if dropped
        else pd.DataFrame(columns=list(raw.columns) + ["drop_rule", "drop_note"])
    )
    drop_df.to_csv(RESULTS / "step1e_dropped_rows.csv", index=False, encoding="utf-8")

    emit("=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    emit(f"{'rows in':<42}{start_n:>7}")
    emit(f"{'  R1 rent / lease':<42}{-n1:>7}")
    emit(f"{'  R2 non-land record':<42}{-n2:>7}")
    emit(f"{'  R3 Title names another district':<42}{-n3:>7}")
    emit(f"{'  R5 scale mislabelled':<42}{-n5:>7}")
    emit(f"{'  R10 per acre (unrecoverable)':<42}{-n10:>7}")
    emit(f"{'  R11 re-posted duplicate advert':<42}{-n11:>7}")
    emit(f"{'total dropped':<42}{-(n1 + n2 + n3 + n5 + n10 + n11):>7}")
    emit(f"{'rows out':<42}{len(df):>7}")
    emit(f"{'  of which price_censored = 1':<42}{n_cens:>7}")
    emit(f"{'GN division names represented':<42}{df.gn_division.nunique():>7}"
         f"  of {n_div_names}")
    emit("")
    emit(f"cleaned dataset : data/interim/listings_gn_clean.csv")
    emit(f"dropped rows    : results/step1e_dropped_rows.csv ({len(drop_df)} rows)")

    (RESULTS / "step1e_cleaning_report.txt").write_text("\n".join(out), encoding="utf-8")

    # -------------------------------------------------- division-level summary
    rows = []
    for gn, g in df.groupby("gn_division"):
        n, med, q1, q3, iqr = iqr_block(g.Price_per_Perch)
        rows.append(
            {
                "gn_division": gn,
                "ds_divisions": lookup.set_index("gn_division").ds_divisions.get(gn, ""),
                "n": n,
                "median_price_per_perch": med,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "min": g.Price_per_Perch.min(),
                "max": g.Price_per_Perch.max(),
                "n_censored": int(g.price_censored.sum()),
                "median_land_size_perch": g.Land_Size_Perch.median(),
                "lat": g.gn_lat.iloc[0],
                "lon": g.gn_lon.iloc[0],
            }
        )
    summary = (
        pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    )
    summary.to_csv(RESULTS / "step1e_division_summary.csv", index=False, encoding="utf-8")

    lines = []
    lines.append("=" * 100)
    lines.append("STEP 1e - PRICE PER PERCH BY GN DIVISION (cleaned dataset)")
    lines.append("=" * 100)
    lines.append(f"rows: {len(df)}   GN division names represented: {len(summary)} of {n_div_names}")
    lines.append(f"({n_div_ds} GN divisions counted within DS division; {n_multi} names span more than one DS)")
    lines.append("")
    lines.append(f"{'GN division':<26}{'n':>6}{'median':>13}{'Q1':>13}{'Q3':>13}{'IQR':>13}{'cens':>6}")
    lines.append("-" * 90)
    for r in summary.itertuples():
        lines.append(
            f"{r.gn_division:<26}{r.n:>6}{r.median_price_per_perch:>13,.0f}"
            f"{r.q1:>13,.0f}{r.q3:>13,.0f}{r.iqr:>13,.0f}{r.n_censored:>6}"
        )
    lines.append("")
    lines.append(f"achievable price range in the cleaned data: "
                 f"Rs {df.Price_per_Perch.min():,.0f} to Rs {df.Price_per_Perch.max():,.0f} per perch")
    (RESULTS / "step1e_division_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(out))
    print(f"\nWrote results/step1e_division_summary.csv ({len(summary)} divisions)")
    print("Wrote results/step1e_division_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

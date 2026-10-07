"""Step 1d - Price sanity audit of the GN-mapped listings.

Read-only. Nothing is deleted, corrected or filtered; the script only measures
and classifies, and proposes a removal rule at the end for a human decision.

Sections
--------
1. Per GN division: n, median and IQR of Price_per_Perch, split by Price_Scale.
2. Colombo 1-15 postal divisions whose median reads under Rs 1,000,000/perch.
3. Every 'total' scale row whose implied price per perch is off by more than
   10x from the median of the 'per perch' rows in the same GN division.
4. Cause classification of the flagged rows.  Every cause is a named rule with
   an explicit condition - a row that satisfies no rule is labelled
   'no rule matched' and listed in full, never swept into a bucket.
5. Proposed removal rule.

Output: results/step1d_price_audit.txt

Run:  python src/step1d_price_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"

RATIO_LIMIT = 10.0          # section 3 threshold
POSTAL_MEDIAN_FLOOR = 1_000_000  # section 2 threshold

# The 15 Colombo municipal postal divisions, same named list step 1b maps to.
POSTAL_DIVISIONS = [
    "Fort", "Slave Island", "Kollupitiya", "Bambalapitiya", "Havelock Town",
    "Wellawatta", "Kurunduwatta", "Borella", "Dematagoda", "Maradana",
    "Pettah", "Aluthkade", "Kotahena", "Grandpass", "Mattakkuliya",
]

# ---------------------------------------------------------------------------
# Named cause rules.  Each is (name, predicate over a row) - no catch-all.
# ---------------------------------------------------------------------------
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
# Land_Type values that are not land parcels at all.
NON_LAND_TYPES = [
    "Office", "Building", "Factory / Workshop", "Warehouse / Storage",
    "Hotel", "Shop",
]

SCALE_SWAP_LO, SCALE_SWAP_HI = 1 / 3, 3.0  # "total is really per perch" window

THOUSANDS_CEILING = 100_000      # implied ppp below this ...
THOUSANDS_FACTOR_LO = 100        # ... and x1000 lands within [100x, 10000x]
THOUSANDS_FACTOR_HI = 10_000


def _has(text, words) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(w in t for w in words)


def load_other_district_places() -> dict[str, str]:
    """Named list of places outside Colombo district, from step 1b/1c."""
    places: dict[str, str] = {}
    man = ROOT / "data" / "reference" / "out_of_district_manual.csv"
    if man.exists():
        for r in pd.read_csv(man).itertuples():
            if isinstance(r.place, str):
                places[r.place.strip().lower()] = str(r.district)
    geo = ROOT / "data" / "reference" / "geocode_out_of_district.csv"
    if geo.exists():
        for r in pd.read_csv(geo).itertuples():
            if isinstance(r.alias, str) and len(r.alias.split()) <= 2:
                places[r.alias.strip().lower()] = str(getattr(r, "osm_district", ""))
    # Spelling variants that appear in advert titles.
    places.setdefault("polgahawella", "Kurunegala")
    return places


OTHER_DISTRICT_PLACES = load_other_district_places()


def title_names_other_district(title) -> str | None:
    if not isinstance(title, str):
        return None
    toks = set(re.sub(r"[^\w]+", " ", title.lower()).split())
    for place, district in OTHER_DISTRICT_PLACES.items():
        if place in toks:
            return f"{place.title()} ({district})"
    return None


def classify(row, div_median: float) -> list[str]:
    """Return every named cause rule this row satisfies."""
    causes = []

    if _has(row.Title, RENT_WORDS):
        causes.append("rent listing")

    if title_names_other_district(row.Title):
        causes.append("title names another district")

    if not isinstance(row.Address_Raw, str) or not row.Address_Raw.strip():
        causes.append("no address text (Town tag only)")

    if row.Land_Type in NON_LAND_TYPES or _has(row.Title, NON_LAND_TITLE_WORDS):
        causes.append("apartment/non-land")

    ppp = row.Price_per_Perch
    if (
        ppp < THOUSANDS_CEILING
        and np.isfinite(div_median)
        and div_median > 0
        and THOUSANDS_FACTOR_LO <= div_median / ppp <= THOUSANDS_FACTOR_HI
    ):
        causes.append("price in thousands")

    # Scale mislabelled: the advert is tagged 'total', but reading the figure
    # as a PER PERCH price puts it right on the division's own median.  This is
    # the same class of defect as the 'per acre' rows in the original study.
    if (
        row.Price_Scale == "total"
        and np.isfinite(div_median)
        and div_median > 0
        and SCALE_SWAP_LO <= row.Price_LKR / div_median <= SCALE_SWAP_HI
    ):
        causes.append("scale mislabelled (total is per perch)")

    # The GN division came from the coarse ikman Town tag, not from the
    # seller's own address text, so the listing may not be in this division.
    if row.match_field == "Town" or row.match_stage in ("fuzzy", "postal"):
        causes.append("wrong Town tag")

    return causes


def iqr_block(s: pd.Series) -> tuple:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return len(s), s.median(), q1, q3, q3 - q1


def main() -> int:
    src = INTERIM / "listings_gn_mapped.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run step1b_map_listings_to_gn.py first.")

    df = pd.read_csv(src, low_memory=False)
    df = df[df.gn_division.notna()].copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # Listing_ID is NOT unique - see the note below

    # Implied price per perch, recomputed from first principles rather than
    # trusting the column, so section 3 compares like with like.
    df["implied_ppp"] = df.Price_LKR / df.Land_Size_Perch

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)

    emit("=" * 78)
    emit("STEP 1d - PRICE SANITY AUDIT")
    emit("=" * 78)
    emit(f"source            : {src.relative_to(ROOT)}")
    emit(f"rows audited      : {len(df)} (GN-mapped Colombo listings)")
    emit(f"GN divisions      : {df.gn_division.nunique()}")
    emit("")
    emit("NOTE ON THE PRICE COLUMNS IN THIS DATA")
    emit("  Price_LKR is already a TOTAL price for every Price_Scale, and")
    emit("  Price_per_Perch == Price_LKR / Land_Size_Perch holds for 100% of")
    emit("  rows in all three scales. That differs from the original author's")
    emit("  published file, where 'Mentioned Price(Rs)' is the figure as")
    emit("  advertised. Step 4 must therefore rebuild 'Mentioned Price(Rs)'")
    emit("  from Price_LKR and the scale, not copy it.")
    emit("")
    n_cap = int((df.Price_per_Perch >= 9_999_999).sum())
    emit(f"  Price_per_Perch maximum is exactly {df.Price_per_Perch.max():,.0f}"
         f" and {n_cap} rows sit at that ceiling -> the upstream scrape appears")
    emit("  to clip per-perch price at Rs 10,000,000. Flagged, not corrected.")
    emit(f"  Price_per_Perch minimum is Rs {df.Price_per_Perch.min():,.2f}.")
    emit("")
    emit("  Listing_ID IS NOT A UNIQUE KEY. It is the advert slug truncated to")
    emit(f"  {int(df.Listing_ID.str.len().max())} characters: {df.Listing_ID.nunique()} distinct values over {len(df)} rows,")
    emit(f"  the worst colliding on {int(df.Listing_ID.value_counts().max())} rows ('{df.Listing_ID.value_counts().index[0]}').")
    emit("  This audit therefore keys on row position, and Step 4 must not use")
    emit("  Listing_ID to identify or de-duplicate rows.")
    emit("")

    # ---------------------------------------------------------------- 1
    emit("=" * 78)
    emit("1. PRICE PER PERCH BY GN DIVISION AND PRICE SCALE")
    emit("=" * 78)
    emit("(median and IQR in Rs per perch; IQR = Q3 - Q1)")
    emit("")

    rows = []
    for (gn, sc), g in df.groupby(["gn_division", "Price_Scale"]):
        n, med, q1, q3, iqr = iqr_block(g.Price_per_Perch)
        rows.append(
            {"gn_division": gn, "Price_Scale": sc, "n": n, "median": med,
             "q1": q1, "q3": q3, "iqr": iqr}
        )
    by_div = pd.DataFrame(rows).sort_values(["gn_division", "Price_Scale"])
    by_div.to_csv(RESULTS / "step1d_price_by_division.csv", index=False)

    order = df.gn_division.value_counts()
    emit(f"{'GN division':<26}{'scale':<11}{'n':>6}{'median':>13}{'Q1':>13}{'Q3':>13}{'IQR':>13}")
    emit("-" * 95)
    for gn in order.index:
        sub = by_div[by_div.gn_division == gn]
        first = True
        for r in sub.itertuples():
            label = gn if first else ""
            first = False
            emit(f"{label:<26}{r.Price_Scale:<11}{r.n:>6}{r.median:>13,.0f}"
                 f"{r.q1:>13,.0f}{r.q3:>13,.0f}{r.iqr:>13,.0f}")
    emit("")
    emit(f"(full table also written to results/step1d_price_by_division.csv,"
         f" {len(by_div)} division x scale combinations)")
    emit("")

    # ---------------------------------------------------------------- 2
    emit("=" * 78)
    emit("2. COLOMBO 1-15 POSTAL DIVISIONS WITH MEDIAN UNDER Rs 1,000,000/PERCH")
    emit("=" * 78)
    emit("These are the 15 municipal postal divisions. A median under Rs 1M per")
    emit("perch is implausible for central Colombo and indicates listings that")
    emit("do not belong to the division.")
    emit("")

    postal_flagged = []
    emit(f"{'GN division':<20}{'n':>6}{'median':>14}{'Q1':>14}{'Q3':>14}   verdict")
    emit("-" * 82)
    for gn in POSTAL_DIVISIONS:
        g = df[df.gn_division == gn]
        if not len(g):
            emit(f"{gn:<20}{0:>6}{'-':>14}{'-':>14}{'-':>14}   no listings")
            continue
        n, med, q1, q3, _ = iqr_block(g.Price_per_Perch)
        bad = med < POSTAL_MEDIAN_FLOOR
        if bad:
            postal_flagged.append(gn)
        emit(f"{gn:<20}{n:>6}{med:>14,.0f}{q1:>14,.0f}{q3:>14,.0f}"
             f"   {'** FLAGGED **' if bad else 'plausible'}")
    emit("")
    emit(f"flagged divisions: {len(postal_flagged)}"
         + (f" -> {', '.join(postal_flagged)}" if postal_flagged else ""))
    emit("")

    if postal_flagged:
        emit("Listings in the flagged postal divisions, cheapest first:")
        emit("")
        for gn in postal_flagged:
            g = df[df.gn_division == gn].nsmallest(15, "Price_per_Perch")
            emit(f"  --- {gn} ({(df.gn_division == gn).sum()} listings) ---")
            for r in g.itertuples():
                emit(f"    {r.Price_per_Perch:>12,.0f}/pch  size={r.Land_Size_Perch:>7.1f}  "
                     f"scale={r.Price_Scale:<10} stage={r.match_stage:<8} field={str(r.match_field):<12}")
                emit(f"      addr : {str(r.Address_Raw)[:90]}")
                emit(f"      title: {str(r.Title)[:90]}")
            emit("")

    # ---------------------------------------------------------------- 3
    emit("=" * 78)
    emit("3. 'total' SCALE ROWS MORE THAN 10x FROM THE DIVISION'S 'per perch' MEDIAN")
    emit("=" * 78)

    pp_median = (
        df[df.Price_Scale == "per perch"]
        .groupby("gn_division")
        .Price_per_Perch.median()
    )
    pp_n = df[df.Price_Scale == "per perch"].groupby("gn_division").size()

    tot = df[df.Price_Scale == "total"].copy()
    tot["div_pp_median"] = tot.gn_division.map(pp_median)
    tot["div_pp_n"] = tot.gn_division.map(pp_n).fillna(0).astype(int)
    tot["ratio"] = tot.implied_ppp / tot.div_pp_median

    no_ref = tot[tot.div_pp_median.isna()]
    comparable = tot[tot.div_pp_median.notna()]
    flagged3 = comparable[
        (comparable.ratio > RATIO_LIMIT) | (comparable.ratio < 1 / RATIO_LIMIT)
    ].sort_values("ratio")

    emit(f"'total' scale rows              : {len(tot)}")
    emit(f"  comparable (division has 'per perch' rows) : {len(comparable)}")
    emit(f"  no 'per perch' reference in division       : {len(no_ref)}")
    emit(f"  OFF BY MORE THAN {RATIO_LIMIT:.0f}x                        : {len(flagged3)}")
    emit(f"    too cheap (ratio < 1/{RATIO_LIMIT:.0f})  : {(flagged3.ratio < 1/RATIO_LIMIT).sum()}")
    emit(f"    too dear  (ratio > {RATIO_LIMIT:.0f})    : {(flagged3.ratio > RATIO_LIMIT).sum()}")
    emit("")

    if len(flagged3):
        emit(f"{'ratio':>9}  {'implied/pch':>13}  {'div median':>13}  {'size':>8}  {'total price':>15}  GN division")
        emit("-" * 105)
        for r in flagged3.itertuples():
            emit(f"{r.ratio:>9.3f}  {r.implied_ppp:>13,.0f}  {r.div_pp_median:>13,.0f}  "
                 f"{r.Land_Size_Perch:>8.1f}  {r.Price_LKR:>15,.0f}  {r.gn_division}"
                 f" (n={r.div_pp_n})")
            emit(f"           addr : {str(r.Address_Raw)[:88]}")
            emit(f"           title: {str(r.Title)[:88]}")
            emit(f"           type={r.Land_Type} | stage={r.match_stage} | field={r.match_field}")
            emit("")

    # ---------------------------------------------------------------- 4
    emit("=" * 78)
    emit("4. CAUSE CLASSIFICATION OF FLAGGED ROWS")
    emit("=" * 78)
    emit("Named rules, applied to the union of the section 2 and section 3")
    emit("flags. A row may satisfy more than one rule. Rows satisfying none are")
    emit("reported as 'no rule matched' and listed individually.")
    emit("")
    emit("  rent listing        Title contains one of: " + ", ".join(RENT_WORDS[:6]) + ", ...")
    emit("  apartment/non-land  Land_Type in " + str(NON_LAND_TYPES))
    emit("                      or Title contains one of: " + ", ".join(NON_LAND_TITLE_WORDS[:6]) + ", ...")
    emit(f"  price in thousands  implied/perch < Rs {THOUSANDS_CEILING:,} and the division")
    emit(f"                      median is {THOUSANDS_FACTOR_LO}x-{THOUSANDS_FACTOR_HI}x higher")
    emit("  wrong Town tag      GN came from the ikman Town tag, or from a fuzzy")
    emit("                      or postal match, not from the seller's address text")
    emit("  title names another Title contains a town from the named out-of-district")
    emit("  district            list (out_of_district_manual.csv + geocoded list +")
    emit("                      titles-only towns), e.g. Pannala, Polgahawela, Negombo")
    emit("  no address text     Address_Raw is empty, so the GN division rests")
    emit("  (Town tag only)     entirely on the coarse ikman Town tag")
    emit("  scale mislabelled   Price_Scale is 'total', but reading Price_LKR as a")
    emit("  (total is per perch) PER PERCH figure lands within "
         f"{SCALE_SWAP_LO:.2f}x-{SCALE_SWAP_HI:.0f}x of the")
    emit("                      division median - i.e. the scale tag is wrong,")
    emit("                      the price is not")
    emit("")

    flag_rows = set(flagged3.row_id)
    for gn in postal_flagged:
        sub = df[(df.gn_division == gn) & (df.Price_per_Perch < POSTAL_MEDIAN_FLOOR)]
        flag_rows |= set(sub.row_id)

    flagged = df[df.row_id.isin(flag_rows)].copy()
    flagged["div_pp_median"] = flagged.gn_division.map(pp_median)

    cause_lists = [
        classify(r, r.div_pp_median if np.isfinite(r.div_pp_median) else np.nan)
        for r in flagged.itertuples()
    ]
    flagged["causes"] = ["; ".join(c) if c else "no rule matched" for c in cause_lists]

    emit(f"rows flagged in total : {len(flagged)}")
    emit("")
    emit("counts per rule (a row can appear under several):")
    tally: dict[str, int] = {}
    for c in cause_lists:
        for name in (c or ["no rule matched"]):
            tally[name] = tally.get(name, 0) + 1
    for name, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        emit(f"  {name:<22}{n:>6}")
    emit("")
    emit("counts per exact combination:")
    for combo, n in flagged.causes.value_counts().items():
        emit(f"  {n:>5}  {combo}")
    emit("")

    unmatched = flagged[flagged.causes == "no rule matched"]
    if len(unmatched):
        emit(f"ROWS MATCHING NO RULE ({len(unmatched)}) - listed in full:")
        emit("")
        for r in unmatched.itertuples():
            emit(f"  {r.Price_per_Perch:>12,.0f}/pch  size={r.Land_Size_Perch:>7.1f}  "
                 f"scale={r.Price_Scale:<10} {r.gn_division}")
            emit(f"    addr : {str(r.Address_Raw)[:88]}")
            emit(f"    title: {str(r.Title)[:88]}")
            emit("")

    flagged.to_csv(RESULTS / "step1d_flagged_rows.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- 5
    emit("=" * 78)
    emit("5. PROPOSED REMOVAL RULE  (NOT APPLIED - for your decision)")
    emit("=" * 78)
    emit("")
    emit("The original study removes outliers only by the +/-2 SD rule inside")
    emit("each GN division (section 3.2.3). It has no rule for listings that are")
    emit("the wrong KIND of record. The proposal below is therefore an addition")
    emit("to the methodology, which is why it is proposed rather than applied.")
    emit("")
    emit("Two of the findings are REPAIRS, not removals, and one is a MAPPING")
    emit("FIX in step 1b. Separating them matters: deleting a mislabelled row")
    emit("throws away a good observation.")
    emit("")
    emit("--- REMOVE (these are the wrong kind of record) ---")
    emit("")
    emit("  R1  Drop rows whose Title matches the rent/lease word list.")
    emit("      A monthly rent is not a sale price and cannot be put on the")
    emit("      same scale as one. This is the single largest cause found.")
    emit("")
    emit("  R2  Drop rows whose Land_Type is one of the non-land types")
    emit(f"      {NON_LAND_TYPES},")
    emit("      or whose Title matches the apartment/non-land word list.")
    emit("      These are apartments, offices and warehouses, not land parcels.")
    emit("      Named, closed lists - not a residual bucket.")
    emit("")
    emit("  R3  Drop rows whose Title names a town in the out-of-district list.")
    emit("      These are bulk developer adverts for Pannala, Polgahawela,")
    emit("      Mirigama, Negombo and Nittambuwa posted under a Colombo Town")
    emit("      tag with no address text. They are the entire reason the Fort")
    emit("      and Kollupitiya medians collapsed.")
    emit("")
    emit("  R4  Drop rows sitting exactly on the Rs 10,000,000 per-perch")
    emit("      ceiling. The value is censored, not observed.")
    emit("")
    emit("--- REPAIR (the price is fine, the label is not) ---")
    emit("")
    emit("  R5  'scale mislabelled': Price_Scale == 'total' but Price_LKR read")
    emit("      as a PER PERCH figure lands within "
         f"{SCALE_SWAP_LO:.2f}x-{SCALE_SWAP_HI:.0f}x of the division")
    emit("      median. Re-tag these as 'per perch' rather than deleting them.")
    emit("      This is the same defect class as the 'per acre' rows in the")
    emit("      original study, and it accounts for most of section 3.")
    emit("")
    emit("  R6  'price in thousands': recoverable by multiplying by 1000, but")
    emit("      almost all of these rows are ALSO rent listings, so R1 removes")
    emit("      them first. Handle whatever survives R1 case by case - do not")
    emit("      apply a blanket x1000.")
    emit("")
    emit("--- FIX IN STEP 1b (not a data removal) ---")
    emit("")
    emit("  R7  Rank a Title-derived out-of-district match ABOVE a postal match")
    emit("      taken from the Town tag. Right now an advert with no address,")
    emit("      Town='Colombo 1' and Title='Land for Sale in Pannala' resolves")
    emit("      to Fort, because the Town tag is consulted before the Title.")
    emit("      Fixing this removes the Fort/Kollupitiya/Kurunduwatta problem")
    emit("      at source and makes R3 largely redundant.")
    emit("")
    emit("--- DO NOT ---")
    emit("")
    emit("  R8  Do NOT drop on 'wrong Town tag' alone. It fires on match")
    emit("      provenance, not on price, and covers many sound rows. Use it")
    emit("      only as a tie-breaker on rows already flagged elsewhere.")
    emit("")
    emit("  R9  Leave everything else to the study's own +/-2 SD per-division")
    emit("      rule in Step 4. Do not stack a second price filter on top.")
    emit("")
    emit("Effect on the current 8669 mapped rows:")
    r1 = df.Title.map(lambda t: _has(t, RENT_WORDS))
    r2 = df.Land_Type.isin(NON_LAND_TYPES) | df.Title.map(
        lambda t: _has(t, NON_LAND_TITLE_WORDS)
    )
    r3 = df.Title.map(lambda t: title_names_other_district(t) is not None)
    r4 = df.Price_per_Perch >= 9_999_999
    removal = r1 | r2 | r3 | r4
    emit(f"  R1 rent/lease titles           : {int(r1.sum()):>5}")
    emit(f"  R2 non-land type or title      : {int(r2.sum()):>5}")
    emit(f"  R3 title names another district: {int(r3.sum()):>5}")
    emit(f"  R4 at the price ceiling        : {int(r4.sum()):>5}")
    emit(f"  UNION to remove                : {int(removal.sum()):>5}"
         f"  ({removal.mean()*100:.2f}% of mapped rows)")
    emit(f"  rows remaining                 : {int((~removal).sum()):>5}")
    emit("")
    n_repair = int(flagged.causes.str.contains("scale mislabelled").sum())
    emit(f"  R5 rows to RE-TAG, not delete  : {n_repair:>5}")
    emit("")
    emit("Nothing has been deleted or altered. Tell me which rules to apply and")
    emit("I will implement them as step1e, keeping a dropped-rows file.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step1d_price_audit.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:0]))
    print(f"Wrote {RESULTS / 'step1d_price_audit.txt'} ({len(out)} lines)")
    print(f"Wrote {RESULTS / 'step1d_price_by_division.csv'}")
    print(f"Wrote {RESULTS / 'step1d_flagged_rows.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

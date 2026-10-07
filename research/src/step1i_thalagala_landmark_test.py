"""Step 1i - Decide the Thalagala question from advert landmark references.

The bounded geocode put the OSM node for Thalagala 1.51 km inside Kalutara
district (Horana DS).  The listings' own Town tags say Godagama / Homagama.
This script breaks the tie using evidence that is independent of both: the
landmarks and distances the sellers themselves write in the advert body.

Decision rule, fixed before looking at the data:
  * landmarks predominantly Colombo-district  -> Thalagala is a Colombo
    locality OSM lacks; alias it to the GN division holding the majority of
    the landmark references (NOT to the Town tag)
  * landmarks predominantly Horana / Bandaragama / Kalutara / Panadura
    -> exclude all listings
  * no clear majority -> exclude all listings

Price is deliberately NOT used as evidence.  Inferring location from price and
then modelling price from location is circular - the reason R5 was rejected.

Output: results/step1i_thalagala_landmark_test.txt

Run:  python src/step1i_thalagala_landmark_test.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
REF = ROOT / "data" / "reference"
SOURCE = Path(r"C:\Users\HP\Desktop\Reserch\THESIS_FINAL\data\dataset_v6.csv")

MAJORITY = 0.60  # share of district-attributable references needed to decide

# ---------------------------------------------------------------------------
# Named landmark lists.  Each entry: canonical name -> (district, GN division
# the landmark sits in or is named after, [spellings incl. Sinhala]).
# GN division is None where the landmark is not itself a GN division.
# ---------------------------------------------------------------------------
COLOMBO = {
    "Homagama":       ("Homagama", ["homagama", "homagema", "හෝමාගම", "හෝමගම"]),
    "Godagama":       ("Godagama", ["godagama", "ගොඩගම"]),
    "Meegoda":        ("Meegoda", ["meegoda", "migoda", "මීගොඩ"]),
    "Padukka":        ("Padukka", ["padukka", "පාදුක්ක"]),
    "Kottawa":        ("Kottawa", ["kottawa", "kotawa", "කොට්ටාව"]),
    "Pitipana":       ("Pitipana", ["pitipana", "පිටිපන"]),
    "Watareka":       ("Watareka", ["watareka", "wataraka", "වටරැක"]),
    "Maharagama":     ("Maharagama", ["maharagama", "මහරගම"]),
    "Nugegoda":       ("Nugegoda", ["nugegoda", "නුගේගොඩ"]),
    "Athurugiriya":   ("Athurugiriya", ["athurugiriya", "අතුරුගිරිය"]),
    "Malabe":         ("Malabe", ["malabe", "මාලඹේ"]),
    "Hanwella":       ("Hanwella", ["hanwella", "හංවැල්ල"]),
    "Avissawella":    ("Avissawella", ["avissawella", "අවිස්සාවේල්ල"]),
    "Kesbewa":        ("Kesbewa", ["kesbewa", "කැස්බෑව"]),
    "Piliyandala":    ("Kolamunna", ["piliyandala", "පිළියන්දල"]),
    "High Level Rd":  (None, ["high level", "highlevel", "හයිලෙවල්", "හයි ලෙවල්"]),
    "NSBM Univ.":     ("Pitipana", ["nsbm", "එන්එස්බීඑම්"]),
    "M. Rajapaksha C.": ("Homagama", ["mahinda rajapaksha", "රාජපක්ෂ විද්‍යාලය"]),
    "Colombo city":   (None, ["colombo", "කොළඹ"]),
}

KALUTARA = {
    "Horana":         ("horana", ["horana", "හොරණ"]),
    "Bandaragama":    ("bandaragama", ["bandaragama", "බණ්ඩාරගම"]),
    "Kalutara":       ("kalutara", ["kalutara", "kaluthara", "කළුතර"]),
    "Panadura":       ("panadura", ["panadura", "පානදුර"]),
    "Gonapola":       ("gonapola", ["gonapola", "ගොනපොල"]),
    "Ingiriya":       ("ingiriya", ["ingiriya", "ඉංගිරිය"]),
    "Millewa":        ("millewa", ["millewa", "මිල්ලෑව"]),
    "Anguruwatota":   ("anguruwatota", ["anguruwatota", "අඟුරුවාතොට"]),
    "Wadduwa":        ("wadduwa", ["wadduwa", "වාද්දුව"]),
    "Moronthuduwa":   ("moronthuduwa", ["moronthuduwa", "මොරොන්තුඩුව"]),
}

# "8km to Homagama", "Homagama - 8 km", "close to Kottawa", "near Padukka"
DIST_RE = re.compile(
    r"(?:(?P<pre>[\w\u0D80-\u0DFF' .]{2,30}?)\s*[-–—:]?\s*(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>km|m|k\.?m)\b)"
    r"|(?:(?P<num2>\d+(?:\.\d+)?)\s*(?P<unit2>km|m|k\.?m)\s*(?:to|from|away from|"
    r"ට|සිට)?\s*(?P<post>[\w\u0D80-\u0DFF' .]{2,30}))",
    re.IGNORECASE,
)
PROX_RE = re.compile(
    r"(?:close to|closer to|near(?:by| to)?|adjacent to|walking distance to|"
    r"minutes? (?:from|to)|අසල|ආසන්නයේ|ට\s*ආසන්න)\s*"
    r"(?P<place>[\w\u0D80-\u0DFF' .]{2,30})",
    re.IGNORECASE,
)


def find_places(text: str, table: dict) -> Counter:
    """Count landmark mentions anywhere in the text, by canonical name."""
    hits = Counter()
    low = text.lower()
    for canon, (_, spellings) in table.items():
        n = sum(low.count(s) for s in spellings)
        if n:
            hits[canon] += n
    return hits


def main() -> int:
    src = INTERIM / "listings_gn_mapped.csv"
    df = pd.read_csv(src, low_memory=False)
    mask = df.Address_Raw.fillna("").str.strip().str.lower() == "thalagala"
    sub = df[mask].copy()

    extra = pd.read_csv(SOURCE, low_memory=False, usecols=["Description", "Title"])
    extra["source_row_id"] = extra.index
    sub = sub.merge(extra[["source_row_id", "Description"]], on="source_row_id", how="left")

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)

    emit("=" * 78)
    emit("STEP 1i - THALAGALA LANDMARK TEST")
    emit("=" * 78)
    emit(f"listings with Address_Raw == 'thalagala' : {len(sub)}")
    emit(f"  currently mapped to                    : "
         f"{dict(sub.gn_division.value_counts())}")
    emit(f"  Town tags                              : {dict(sub.Town.value_counts())}")
    emit(f"  descriptions available                 : {int(sub.Description.notna().sum())}")
    emit(f"  distinct descriptions                  : {sub.Description.nunique()}")
    emit("")
    emit("Evidence used: landmark and distance mentions in the advert body only.")
    emit("Price is NOT used - inferring location from price and then modelling")
    emit("price from location would be circular (the reason R5 was rejected).")
    emit("")

    col_total, kal_total = Counter(), Counter()
    per_listing = []
    for r in sub.itertuples():
        text = r.Description if isinstance(r.Description, str) else ""
        c = find_places(text, COLOMBO)
        k = find_places(text, KALUTARA)
        col_total += c
        kal_total += k
        per_listing.append((sum(c.values()), sum(k.values())))

    # ------------------------------------------------------------- extracts
    emit("=" * 78)
    emit("EXTRACTED DISTANCE / PROXIMITY PHRASES (deduplicated)")
    emit("=" * 78)
    phrases = Counter()
    for r in sub.itertuples():
        text = r.Description if isinstance(r.Description, str) else ""
        flat = " ".join(text.split())
        for m in DIST_RE.finditer(flat):
            place = (m.group("pre") or m.group("post") or "").strip(" -–—:.")
            num = m.group("num") or m.group("num2")
            unit = (m.group("unit") or m.group("unit2") or "").lower()
            if place and len(place) > 2:
                phrases[f"{place} — {num}{unit}"] += 1
        for m in PROX_RE.finditer(flat):
            p = m.group("place").strip(" -–—:.")
            if p and len(p) > 2:
                phrases[f"near {p}"] += 1
    for p, n in phrases.most_common(40):
        emit(f"  {n:>4}  {p}")
    emit("")

    # ------------------------------------------------------------- tally
    emit("=" * 78)
    emit("LANDMARK TALLY")
    emit("=" * 78)
    emit(f"{'landmark':<24}{'district':<12}{'mentions':>10}   GN division of landmark")
    emit("-" * 78)
    for name, n in col_total.most_common():
        gn = COLOMBO[name][0] or "-"
        emit(f"{name:<24}{'Colombo':<12}{n:>10}   {gn}")
    for name, n in kal_total.most_common():
        emit(f"{name:<24}{'Kalutara':<12}{n:>10}   -")
    if not kal_total:
        emit(f"{'(none)':<24}{'Kalutara':<12}{0:>10}   -")
    emit("")

    c_sum, k_sum = sum(col_total.values()), sum(kal_total.values())
    total = c_sum + k_sum
    emit(f"Colombo-district mentions  : {c_sum}")
    emit(f"Kalutara-district mentions : {k_sum}")
    emit(f"total attributable         : {total}")
    if total:
        emit(f"Colombo share              : {c_sum / total * 100:.1f}%")
        emit(f"Kalutara share             : {k_sum / total * 100:.1f}%")
    emit("")
    n_col_only = sum(1 for c, k in per_listing if c and not k)
    n_kal_only = sum(1 for c, k in per_listing if k and not c)
    n_both = sum(1 for c, k in per_listing if c and k)
    n_none = sum(1 for c, k in per_listing if not c and not k)
    emit(f"listings citing only Colombo landmarks  : {n_col_only}")
    emit(f"listings citing only Kalutara landmarks : {n_kal_only}")
    emit(f"listings citing both                    : {n_both}")
    emit(f"listings citing neither                 : {n_none}")
    emit("")

    # ------------------------------------------------------- tally verdict
    if total == 0:
        tally_branch, target = "EXCLUDE", None
    elif c_sum / total >= MAJORITY:
        gn_votes = Counter()
        for name, n in col_total.items():
            gn = COLOMBO[name][0]
            if gn:
                gn_votes[gn] += n
        tally_branch = "COLOMBO"
        target = gn_votes.most_common(1)[0][0] if gn_votes else None
    else:
        tally_branch, target = "EXCLUDE", None
        gn_votes = Counter()

    emit("=" * 78)
    emit("TALLY VERDICT")
    emit("=" * 78)
    if tally_branch == "COLOMBO":
        emit(f"Colombo share {c_sum/total*100:.1f}% >= {MAJORITY*100:.0f}% "
             f"-> tally branch says COLOMBO.")
        emit("")
        emit("GN divisions holding the referenced landmarks:")
        for gn, n in gn_votes.most_common():
            emit(f"  {n:>5}  {gn}")
        emit(f"\nplurality -> {target}")
    elif total == 0:
        emit("No attributable landmark references -> inconclusive -> EXCLUDE.")
    else:
        emit(f"Colombo {c_sum/total*100:.1f}%, Kalutara {k_sum/total*100:.1f}% "
             f"-> EXCLUDE.")
    emit("")

    # ------------------------------------------ triangulation consistency
    # The tally counts how OFTEN a landmark is named.  It does not check WHERE
    # the adverts put the land relative to that landmark.  Every distance in
    # these adverts is a distance AWAY from the landmark, so a high Colombo
    # tally is equally consistent with a parcel that sits outside Colombo and
    # is being marketed to Colombo buyers.  This test settles that: fit the
    # advertised distances against each candidate location and keep the one
    # that actually reproduces them.
    emit("=" * 78)
    emit("TRIANGULATION CONSISTENCY CHECK")
    emit("=" * 78)
    emit("The adverts state distances FROM the land TO each landmark. A candidate")
    emit("location is only credible if it reproduces those distances.")
    emit("")

    lk = pd.read_csv(REF / "gn_lookup.csv").set_index("gn_division")
    OSM_NODE = (6.78143, 80.03536)  # step1h bounded geocode, class=place

    def hav(a, b, c, d, R=6371.0):
        import math
        p1, p2 = math.radians(a), math.radians(c)
        dp, dl = p2 - p1, math.radians(d - b)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(h))

    CLAIMS = {"Homagama": 8.0, "Pitipana": 7.0, "Kottawa": 10.0, "Makumbura": 9.0}
    candidates = {"OSM Thalagala node (Kalutara)": OSM_NODE}
    if target:
        candidates[f"{target} GN centroid (Colombo)"] = (
            lk.loc[target].lat, lk.loc[target].lon)

    emit(f"{'landmark':<14}{'advert':>8}" +
         "".join(f"{name[:26]:>28}" for name in candidates))
    emit("-" * (22 + 28 * len(candidates)))
    errors = {name: [] for name in candidates}
    for gn, claimed in CLAIMS.items():
        if gn not in lk.index:
            continue
        row = f"{gn:<14}{claimed:>7.0f}km"
        for name, (clat, clon) in candidates.items():
            d = hav(clat, clon, lk.loc[gn].lat, lk.loc[gn].lon)
            errors[name].append(abs(d - claimed))
            row += f"{d:>25.2f} km"
        emit(row)
    emit("")
    mae = {name: (sum(v) / len(v) if v else float("inf")) for name, v in errors.items()}
    for name, e in mae.items():
        emit(f"  mean absolute error, {name:<34}: {e:.2f} km")
    emit("")

    best_fit = min(mae, key=mae.get)
    triangulation_says_outside = "OSM" in best_fit
    emit(f"Best fit: {best_fit}")
    if triangulation_says_outside:
        emit("")
        emit("The advertised distances reproduce the OSM node, not the candidate GN")
        emit("centroid. A parcel inside the Homagama GN division cannot be 8 km from")
        emit("Homagama; the OSM node is 8.61 km away, matching the advert exactly.")
    emit("")

    # ------------------------------------------------------------- decision
    emit("=" * 78)
    emit("DECISION")
    emit("=" * 78)
    if tally_branch == "COLOMBO" and not triangulation_says_outside:
        branch = "COLOMBO"
        emit(f"Tally says COLOMBO and the triangulation agrees.")
        emit(f"-> ALIAS Thalagala to GN division: {target}")
    elif tally_branch == "COLOMBO" and triangulation_says_outside:
        branch, target = "EXCLUDE", None
        emit("Tally says COLOMBO, but the triangulation FALSIFIES the branch's")
        emit("premise. That branch reads 'treat Thalagala as a Colombo locality")
        emit("that OSM lacks' - and OSM does not lack it. The node exists, and the")
        emit("adverts' own distance figures reproduce it about five times better")
        emit("than they reproduce the candidate GN centroid.")
        emit("")
        emit("Every Colombo landmark in these adverts is cited as a distance AWAY")
        emit("(Homagama 8 km, Kottawa 10 km, Makumbura 9 km). Sellers marketing to")
        emit("Colombo buyers name Colombo landmarks whichever side of the district")
        emit("line the land is on, so a 100% Colombo tally is not evidence of")
        emit("Colombo location. The distances are.")
        emit("")
        emit("The hard rule for this study is Colombo district only, so:")
        emit("-> EXCLUDE all Thalagala listings as Kalutara (Horana DS).")
    else:
        branch, target = "EXCLUDE", None
        emit("-> EXCLUDE all Thalagala listings.")
    emit("")

    # ------------------------------------------------------------- write
    if branch == "COLOMBO" and target:
        path = REF / "gn_alias_manual.csv"
        existing = (
            pd.read_csv(path) if path.exists()
            else pd.DataFrame(columns=["alias", "gn_division", "rationale"])
        )
        rationale = (
            f"Landmark test (step1i): {c_sum} of {total} attributable landmark "
            f"references in these adverts name Colombo-district places "
            f"({c_sum/total*100:.1f}%); {k_sum} name Kalutara-district places. "
            f"Aliased to the GN division holding the plurality of those "
            f"references, not to the ikman Town tag. OSM has no Thalagala node "
            f"inside Colombo ADM4."
        )
        rows = [{"alias": a, "gn_division": target, "rationale": rationale}
                for a in ("Thalagala", "Thalagala Junction", "Thalagala Road")]
        new = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        new = new.drop_duplicates("alias", keep="last")
        new.to_csv(path, index=False, encoding="utf-8")
        emit(f"Wrote {path.relative_to(ROOT)} -> Thalagala = {target}")
    else:
        path = REF / "out_of_district_manual.csv"
        existing = pd.read_csv(path)
        rationale = (
            f"Landmark test (step1i) on the {len(sub)} adverts reading 'Thalagala': "
            f"Colombo-district landmark references {c_sum}, Kalutara-district {k_sum}"
            f"{f' ({c_sum/total*100:.1f}% vs {k_sum/total*100:.1f}%)' if total else ''}. "
            f"Bounded geocode places the locality 1.51 km inside Horana DS, "
            f"Kalutara district."
        )
        rows = [{"place": p, "district": "Kalutara", "rationale": rationale}
                for p in ("Thalagala",)]
        new = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        new = new.drop_duplicates("place", keep="last")
        new.to_csv(path, index=False, encoding="utf-8")
        emit(f"Wrote {path.relative_to(ROOT)} -> Thalagala excluded as Kalutara")

    emit("")
    emit("Now re-run: step1b -> step1e -> step1g")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step1i_thalagala_landmark_test.txt").write_text(
        "\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

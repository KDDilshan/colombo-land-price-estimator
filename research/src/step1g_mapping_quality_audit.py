"""Step 1g - R13 and R14 mapping-quality audits.  READ-ONLY.

R13  Audit the rows that fell back to the coarse ikman Town tag even though
     the seller supplied address text.  Every distinct unmatched string is
     listed with its count and classified by a NAMED rule:

       out-of-district town      the string names a town outside Colombo that
                                 the R12 lists do not yet carry
       GN name misspelt          the best fuzzy candidate among GN division
                                 names scores in [NEAR_LO, FUZZY_CUTOFF) -
                                 a near miss below the acceptance threshold
       road or landmark only     every token is generic address vocabulary or
                                 a thoroughfare word - there is no place name
                                 in the string at all
       listing code only         the string is an agent reference such as
                                 LS-628 or PILIYANDALA-601LS with no usable
                                 place token beyond one already tried
       no rule matched           listed individually, never bucketed

R14  Determinism: after R12, no distinct Address_Raw string may map to more
     than one GN division, except where Address_Raw is blank.

Output: results/step1g_mapping_quality_audit.txt
        results/step1g_town_fallback_strings.csv
        results/step1g_nondeterministic_addresses.csv

Run:  python src/step1g_mapping_quality_audit.py
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1b_map_listings_to_gn import (  # noqa: E402
    FUZZY_CUTOFF,
    NON_PLACE_WORDS,
    THOROUGHFARE_WORDS,
    GNIndex,
    latin_key,
    sound_key,
    tokenize,
)

INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"

NEAR_LO = 0.70          # below this, "misspelt" is not a credible claim
CODE_RE = re.compile(r"(?:^|[\s\-])(?:ls|dsp|af|afa|pr|nk|em|cl|u|m|s|k|r)[\s\-]?\d{2,4}",
                     re.IGNORECASE)


def best_gn_candidate(text: str, idx: GNIndex):
    """Best near-miss GN division name for a string, below the accept cutoff."""
    toks = [t for t in tokenize(text) if re.search(r"[a-zA-Z]", t)]
    toks = [t for t in toks if t.lower() not in NON_PLACE_WORDS]
    best = (0.0, None, None)
    for n in (2, 1):
        for i in range(len(toks) - n + 1):
            gram = " ".join(sound_key(t) for t in toks[i : i + n])
            if len(gram) < 5:
                continue
            close = difflib.get_close_matches(gram, idx.fuzzy_pool, n=1, cutoff=NEAR_LO)
            if not close:
                continue
            ratio = difflib.SequenceMatcher(None, gram, close[0]).ratio()
            if ratio > best[0]:
                gn = sorted(idx.key_index[close[0]])[0]
                best = (ratio, gn, " ".join(toks[i : i + n]))
    return best


def classify(text: str, idx: GNIndex, geo: dict) -> tuple[str, str]:
    """Return (rule name, evidence).

    Geocoding is the primary discriminator, because text similarity cannot
    tell a misspelling from a real place that simply is not a GN division:
    "Thalagala" scores 0.875 against the GN name "Malagala" but is a genuine
    locality near Homagama, and "Kindelpitiya" scores 0.727 against
    "Kollupitiya" while being a real place in Kesbewa.
    """
    toks = [t for t in tokenize(text) if re.search(r"[a-zA-Z]", t)]

    ood = idx.ood_priority
    for t in toks:
        if sound_key(t) in ood:
            return "out-of-district town", f"{t} -> {ood[sound_key(t)]}"

    g = geo.get(text.lower().strip())
    if g and g.get("gn"):
        ratio, gn, span = best_gn_candidate(text, idx)
        if gn == g["gn"] and ratio >= 0.80:
            return ("GN name misspelt",
                    f"{span!r} ~ {gn} ({ratio:.3f}, below cutoff {FUZZY_CUTOFF}); "
                    f"geocode agrees")
        return ("real locality, not a GN name",
                f"geocodes inside GN division {g['gn']} ({g['ds']})")
    if g and g.get("district"):
        return ("out-of-district town (low confidence)",
                f"no Colombo-bounded hit; unbounded geocode -> {g['district']}")

    if toks and all(
        t.lower() in NON_PLACE_WORDS or t.lower() in THOROUGHFARE_WORDS for t in toks
    ):
        return "road or landmark only", "every token is generic address vocabulary"

    ratio, gn, span = best_gn_candidate(text, idx)
    if gn is not None and 0.80 <= ratio < FUZZY_CUTOFF:
        return ("GN name misspelt",
                f"{span!r} ~ {gn} ({ratio:.3f}, below cutoff {FUZZY_CUTOFF}); no geocode")

    if CODE_RE.search(text) or not toks:
        return "listing code only", "agent reference, no usable place token"

    place_toks = [t for t in toks if t.lower() not in NON_PLACE_WORDS
                  and t.lower() not in THOROUGHFARE_WORDS]
    if not place_toks:
        return "road or landmark only", "no token outside generic address vocabulary"

    return ("genuinely unresolvable",
            f"no geocode; best GN candidate {gn} at {ratio:.3f}" if gn
            else "no geocode, no GN candidate")


def geocode_strings(strings: list[str]) -> dict:
    """Geocode each unmatched string and locate it against Colombo ADM4.

    TWO queries per string, in this order:

      1. bounded to the Colombo district bounding box (viewbox + bounded=1).
         A hit that lands inside a GN polygon is decisive: the place exists in
         Colombo district.
      2. only if (1) returns nothing, an unbounded Sri Lanka query.  A hit
         here means the name exists elsewhere but not in Colombo.

    The bounded query is essential.  An unbounded query on a bare village name
    returns the top match anywhere in Sri Lanka, which sent "Thalagala" to
    Kalutara, "Palawatta" to Badulla and "colombo area" to Kalutara in the
    first version of this audit.  Verdicts from query (2) are marked
    low-confidence because a road or landmark name can still match anything.

    Cached in data/reference/step1g_geocode_cache.json so re-runs are free.
    """
    import json
    import time

    import requests

    from step1c_geocode_unresolved import build_polygon_cache, locate

    cache_path = ROOT / "data" / "reference" / "step1g_geocode_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    todo = [s for s in strings if s.lower().strip() not in cache]
    if not todo:
        return cache

    polys = build_polygon_cache()
    xs = [p["bbox"][0] for p in polys] + [p["bbox"][2] for p in polys]
    ys = [p["bbox"][1] for p in polys] + [p["bbox"][3] for p in polys]
    viewbox = f"{min(xs)},{max(ys)},{max(xs)},{min(ys)}"

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "colombo-land-price-replication/1.0 (MSc replication)"}
    )
    url = "https://nominatim.openstreetmap.org/search"

    def query(q: str, bounded: bool):
        params = {"q": f"{q}, Sri Lanka", "format": "json", "limit": 5,
                  "countrycodes": "lk", "addressdetails": 1}
        if bounded:
            params.update({"viewbox": viewbox, "bounded": 1})
        try:
            r = session.get(url, params=params, timeout=45)
            return r.json() if r.status_code == 200 else []
        except Exception as exc:
            print(f"    ! {q!r}: {exc}")
            return []

    print(f"geocoding {len(todo)} strings, bounded-first (~{len(todo)*2.2/60:.1f} min) ...")
    for i, s in enumerate(todo, 1):
        entry: dict = {}
        for hit in query(s, bounded=True):
            poly = locate(float(hit["lon"]), float(hit["lat"]), polys)
            if poly:
                entry = {"gn": poly["base_name"], "ds": poly["ds"],
                         "display": hit.get("display_name", ""), "how": "bounded"}
                break
        time.sleep(1.1)
        if not entry:
            hits = query(s, bounded=False)
            time.sleep(1.1)
            if hits:
                top = hits[0]
                addr = top.get("address") or {}
                entry = {"district": addr.get("state_district") or addr.get("county")
                         or "outside Colombo",
                         "display": top.get("display_name", ""),
                         "how": "unbounded (low confidence)"}
        cache[s.lower().strip()] = entry
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def main() -> int:
    src = INTERIM / "listings_gn_mapped.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run step1b_map_listings_to_gn.py first.")

    idx = GNIndex()
    df = pd.read_csv(src, low_memory=False)
    m = df[df.gn_division.notna()].copy()

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)

    emit("=" * 78)
    emit("STEP 1g - R13 / R14 MAPPING QUALITY AUDIT (read-only)")
    emit("=" * 78)
    emit(f"source                  : {src.relative_to(ROOT)}")
    emit(f"listings tagged Colombo : {len(df)}")
    emit(f"mapped to a GN division : {len(m)}")
    emit(f"excluded out-of-district: {int((df.match_stage == 'out_of_district').sum())}")
    emit("")

    # ---------------------------------------------------------------- R13
    tf = m[m.match_field == "Town"].copy()
    tf["addr"] = tf.Address_Raw.fillna("").str.strip()
    blank = tf[tf.addr == ""]
    withtext = tf[tf.addr != ""]

    emit("=" * 78)
    emit("R13 - TOWN-FALLBACK ROWS THAT HAVE ADDRESS TEXT")
    emit("=" * 78)
    emit(f"matched via the Town tag        : {len(tf)}")
    emit(f"  blank Address_Raw (accepted)  : {len(blank)}")
    emit(f"  HAS Address_Raw text          : {len(withtext)}")
    emit("")

    grouped = (
        withtext.groupby(withtext.addr.str.lower())
        .agg(n=("addr", "size"), example=("addr", "first"),
             towns=("Town", lambda s: "/".join(sorted(set(s))[:3])),
             gns=("gn_division", lambda s: "/".join(sorted(set(s)))))
        .sort_values("n", ascending=False)
    )
    emit(f"distinct unmatched address strings : {len(grouped)}")
    emit("")

    geo = geocode_strings([r["example"] for _, r in grouped.iterrows()])

    rows = []
    for key, r in grouped.iterrows():
        rule, evidence = classify(r["example"], idx, geo)
        g = geo.get(r["example"].lower().strip()) or {}
        rows.append({"address_raw": r["example"], "n_listings": int(r["n"]),
                     "rule": rule, "evidence": evidence,
                     "geocoded_gn": g.get("gn", ""), "geocoded_district": g.get("district", ""),
                     "mapped_to": r["gns"], "town_tag": r["towns"]})
    cls = pd.DataFrame(rows)
    cls.to_csv(RESULTS / "step1g_town_fallback_strings.csv", index=False, encoding="utf-8")

    emit("classification summary (strings / listings):")
    summ = cls.groupby("rule").agg(strings=("rule", "size"), listings=("n_listings", "sum"))
    summ = summ.sort_values("listings", ascending=False)
    for rule, r in summ.iterrows():
        emit(f"  {rule:<24}{int(r['strings']):>6} strings{int(r['listings']):>7} listings")
    emit("")

    for rule in summ.index:
        sub = cls[cls.rule == rule].sort_values("n_listings", ascending=False)
        emit("-" * 78)
        emit(f"{rule.upper()}  ({len(sub)} strings, {int(sub.n_listings.sum())} listings)")
        emit("-" * 78)
        for r in sub.itertuples():
            emit(f"{r.n_listings:>5}  {r.address_raw[:52]:<54} -> {r.mapped_to[:28]}")
            emit(f"       town tag: {r.town_tag[:30]:<32} {r.evidence[:60]}")
        emit("")

    # ---------------------------------------------------------------- R14
    emit("=" * 78)
    emit("R14 - DETERMINISM OF Address_Raw -> GN DIVISION")
    emit("=" * 78)
    emit("Requirement: after R12, no distinct Address_Raw string may map to")
    emit("more than one GN division, except where Address_Raw is blank.")
    emit("")

    ha = m[m.Address_Raw.fillna("").str.strip() != ""].copy()
    ha["akey"] = ha.Address_Raw.str.strip().str.lower()
    g = ha.groupby("akey").gn_division.nunique()
    bad = g[g > 1]
    bad_rows = ha[ha.akey.isin(bad.index)]

    emit(f"rows with Address_Raw text            : {len(ha)}")
    emit(f"distinct Address_Raw strings          : {ha.akey.nunique()}")
    emit(f"strings mapping to >1 GN division     : {len(bad)}")
    emit(f"rows involved                         : {len(bad_rows)}")
    emit("")

    if len(bad):
        emit("VIOLATIONS:")
        emit("")
        detail = []
        for key in bad.sort_values(ascending=False).index:
            sub = bad_rows[bad_rows.akey == key]
            emit(f"  {key[:60]!r}  ({len(sub)} rows, {sub.gn_division.nunique()} divisions)")
            for gn, n in sub.gn_division.value_counts().items():
                stages = "/".join(sorted(set(sub[sub.gn_division == gn].match_stage)))
                fields = "/".join(sorted(set(sub[sub.gn_division == gn].match_field)))
                towns = "/".join(sorted(set(sub[sub.gn_division == gn].Town.fillna("<NA>")))[:3])
                emit(f"      {n:>4}  {gn:<22} via {fields}/{stages}   town={towns[:34]}")
                detail.append({"address_raw": key, "gn_division": gn, "n": n,
                               "match_field": fields, "match_stage": stages,
                               "town_tags": towns})
            emit("")
        pd.DataFrame(detail).to_csv(
            RESULTS / "step1g_nondeterministic_addresses.csv", index=False, encoding="utf-8"
        )
        emit("Cause: every violation is a string that failed to match, so the GN")
        emit("division came from the ikman Town tag - and the same string is")
        emit("posted under several different Town tags.  Fixing the strings in")
        emit("R13 removes the violation at source.")
    else:
        emit("NO VIOLATIONS - the mapping is deterministic on Address_Raw.")
        pd.DataFrame(columns=["address_raw", "gn_division", "n"]).to_csv(
            RESULTS / "step1g_nondeterministic_addresses.csv", index=False, encoding="utf-8"
        )

    (RESULTS / "step1g_mapping_quality_audit.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote results/step1g_mapping_quality_audit.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

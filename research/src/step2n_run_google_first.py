"""Step 2n - Google-first, address-anchored, single-pass geocoding.

WHAT CHANGES FROM STEP 2/2j, AND WHY.

  Google is tier 1, not tier 5.  Every source in this batch carries a street
  address, and Nominatim has no house-number geocoding for Colombo - proved in
  NOTES.md 3A.2 with four addresses taken verbatim from the source registers,
  zero hits each.  Running four tiers that cannot succeed before the one that
  can is latency, not rigour.  Licensing exposure is unchanged in kind and
  larger in degree; every affected row still carries coord_source="google" and
  NOTES.md 3A.3 still applies.

  No anchor pre-pass.  THE ADDRESS IS THE ANCHOR.  The Nominatim anchor pass
  existed to bound units known only by a locality; for an address it adds a
  round trip and a known failure mode (16 locality strings collapsing onto the
  Colombo city centroid, NOTES.md 3A.1 form 4).  For units whose location is a
  bare town - railway stations, Ceypetco filling stations - the anchor is that
  town and the acceptance radius still does real work.

  The scope filter runs BEFORE geocoding.  An address naming no Colombo or
  buffer place is dropped unresolved (step2m).  A Kandy branch does not need a
  geocoder to establish that it is in Kandy.

  Source-published coordinates are preferred to any geocoder.  Sampath, NDB,
  Seylan, LOLC, People's Leasing and Commercial Credit publish a coordinate per
  branch.  The institution's own pin is better evidence than a geocoder's
  guess, costs nothing and carries no Google licensing exposure; those rows are
  tagged coord_source="source_published".

  One pass.  No retries, no manual tier, no chasing failures.

GUARDS THAT STAY ON, all of them:
  class/quality gate  google_accept() - a `route` can never locate an
                      institution, and APPROXIMATE is Google's own flag for a
                      town-level centroid
  acceptance radius   ACCEPT_KM per category, measured against the address
                      anchor (degenerate for address-anchored units by
                      construction, real for town-anchored ones)
  duplicate pin       two units may not share a pin within 100 m, with the
                      shared-token exception for related campuses/branches
  Sri Lanka bbox      every coordinate, whatever its source
  coord_source        recorded per row

Run:  python src/step2n_run_google_first.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import step2_geocode_amenities as G  # noqa: E402

REF = ROOT / "data" / "reference"
RESULTS = ROOT / "results"

# Google `types` that can legitimately be one of these units.  Extends the
# base list in step2_geocode_amenities, which was written for hospitals,
# schools and universities only.
G.GOOGLE_RESULT_TYPES_OK |= {
    "bank", "finance", "atm", "insurance_agency", "accounting",
    "gas_station", "supermarket", "grocery_or_supermarket", "store",
    "convenience_store", "department_store", "shopping_mall",
    "train_station", "transit_station", "light_rail_station", "subway_station",
    "doctor", "dentist", "pharmacy", "physiotherapist", "veterinary_care",
    "secondary_school", "general_contractor",
}

ACCEPT = {
    "govtschools": 2.0,
    "banks": 2.0,
    "finance_companies": 2.0,
    "supermarkets": 2.0,
    "railway_stations": 2.0,
    "pvt_med_centers": 2.0,
    "fuel_stations": 2.0,
}
G.ACCEPT_KM.update(ACCEPT)

# THE RADIUS DEPENDS ON WHAT THE ANCHOR IS.
# When the source gives a street address, the anchor is that address and the
# radius is a formality.  When it gives a bare town or postal area - "KADUWELA",
# "Colombo 07" - the anchor is a town centroid, and a filling station or school
# genuinely sits some kilometres from the centre of its own town.  Measuring
# both against 2 km rejected correct pins: Ceypetco Moratuwa at 6.6 km from the
# centre of Moratuwa is in Moratuwa.  A town anchor therefore gets a town-sized
# radius.
TOWN_ANCHOR_KM = 5.0

# The shared-pin exception exists for genuine second campuses (Wycherley Senior
# and Junior).  It keys on shared distinguishing tokens, which is wrong for
# these categories: every Bank of Ceylon branch shares "Ceylon" with every
# other, so the exception swallowed the guard - 81 bank rows and 9 filling
# stations kept a pin already claimed by a different unit.  None of these
# categories has a second-campus case, so no unit may share a pin with another.
SHARED_PIN_ALLOWED: set[str] = set()

# Sri Lanka Railways publishes ROUTE distance from Fort along the track;
# haversine measures a straight line, which is always shorter.  Comparing the
# two symmetrically at 3 km rejected correct pins wholesale - Homagama is
# 20.3 km straight and 26.5 km by rail.  The guard is therefore one-sided: a
# straight line may fall well short of the route distance but may not exceed
# it, because no station is further from Fort in a straight line than along the
# track.
RAIL_OVERSHOOT_KM = 3.0
RAIL_MIN_RATIO = 0.5

# name -> (lat, lon) published by the source itself, per category.
PUBLISHED: dict[str, dict[str, tuple]] = {}

SCOPE_MIN_TOKEN = 4


def load_scope() -> list[str]:
    doc = json.loads((REF / "colombo_scope_placenames.json").read_text(encoding="utf-8"))
    return [t for t in doc["tokens"] if len(t) >= SCOPE_MIN_TOKEN]


def scope_matcher(tokens: list[str]):
    """A unit is in scope if its address names any in-scope place.

    Word-boundary matching, so "Colombo" does not match inside another word and
    a GN name is not matched as a substring of an unrelated one.
    """
    pat = re.compile(r"\b(" + "|".join(sorted((re.escape(t) for t in tokens),
                                              key=len, reverse=True)) + r")\b")

    def hit(address: str) -> str | None:
        a = " ".join(str(address or "").replace("-", " ").split()).lower()
        m = pat.search(a)
        return m.group(1) if m else None
    return hit


# --------------------------------------------------------------------- resolve
def make_resolver(published: dict):
    """A drop-in replacement for G.resolve: Google first, address as anchor."""

    def resolve(session, name, locality, cat, cache, claimed, collapsed=None):
        key = f"GF2|{cat}|{name}|{locality}"
        if key in cache and cache[key].get("status") == "ok":
            r = cache[key]
            if not duplicate_blocked(r["lat"], r["lon"], name, claimed, cat):
                return r

        limit = (G.ACCEPT_KM[cat] if G.is_street_address(locality)
                 else TOWN_ANCHOR_KM)
        rejects: list[str] = []

        def finish(lat, lon, src, disp, km, query):
            if not G.in_sri_lanka(lat, lon):
                rejects.append(f"{src} -> outside the Sri Lanka bounding box")
                return None
            dup = duplicate_of(lat, lon, name, claimed, cat)
            if dup and dup[0] == "blocked":
                rejects.append(f"{src} -> pin claimed by unrelated {dup[1]!r}")
                return None
            if not fort_ok(lat, lon, src, cat, name, rejects):
                return None
            res = {"status": "ok",
                   "coord_source": "shared_campus_pin" if dup else src,
                   "anchor_status": "ok", "anchor_fragment": locality,
                   "lat": lat, "lon": lon, "osm_class": src,
                   "display_name": disp, "km_from_locality": km,
                   "query_used": query, "rejected": rejects[:6]}
            cache[key] = res
            G.save_cache(cache)
            return res

        # ---- tier 0: the source's own coordinate ---------------------------
        pub = published.get(name)
        if pub:
            out = finish(pub[0], pub[1], "source_published",
                         f"[SOURCE-PUBLISHED] {locality}", None, name)
            if out:
                return out

        # ---- tier 1: Google, the address as both query and anchor ----------
        gg = G.google_lookup(session, name, locality, cat, cache)
        if gg:
            anchor = G.google_anchor(session, locality, cache)
            km = (round(G.haversine(anchor[0], anchor[1], gg["lat"], gg["lon"]), 2)
                  if anchor else None)
            if km is not None and km > limit:
                rejects.append(f"google -> {km}km from the address anchor "
                               f"(>{limit}km)")
            else:
                out = finish(gg["lat"], gg["lon"], "google",
                             f"[GOOGLE {gg.get('location_type')}] "
                             f"{gg.get('formatted', '')} "
                             f"types={','.join(gg.get('types', []))}",
                             km, gg.get("query_used", name))
                if out:
                    return out

        res = {"status": "failed", "coord_source": "FAILED",
               "anchor_status": "ok", "anchor_fragment": locality,
               "reason": "; ".join(rejects) or "Google returned no admissible result",
               "rejected": rejects[:10]}
        cache[key] = res
        G.save_cache(cache)
        return res

    return resolve


def distinctive_of(s: str) -> set:
    return {t for t in re.findall(r"[A-Za-z]+", s.lower())
            if t not in G.GENERIC_TOKENS}


def duplicate_of(lat, lon, name, claimed, cat=None):
    mine = distinctive_of(name)
    for (clat, clon), owner in claimed.items():
        if owner == name:
            continue
        if G.haversine(clat, clon, lat, lon) * 1000 > G.DUPLICATE_M:
            continue
        if cat in SHARED_PIN_ALLOWED and mine & distinctive_of(owner):
            return ("shared", owner)
        return ("blocked", owner)
    return None


def duplicate_blocked(lat, lon, name, claimed, cat=None):
    d = duplicate_of(lat, lon, name, claimed, cat)
    return bool(d) and d[0] == "blocked"


def fort_ok(lat, lon, tag, cat, name, rejects):
    """One-sided: straight-line distance may not exceed the published route."""
    if cat != "railway_stations":
        return True
    stated = G.RAILWAY_KM.get(name)
    if stated is None:
        return True
    got = G.haversine(G.FORT[0], G.FORT[1], lat, lon)
    if got > stated + RAIL_OVERSHOOT_KM:
        rejects.append(f"{tag} -> {got:.1f}km straight from Fort, source route "
                       f"is {stated:.1f}km (a straight line cannot be longer)")
        return False
    if stated > 4 and got < RAIL_MIN_RATIO * stated:
        rejects.append(f"{tag} -> {got:.1f}km straight from Fort against a "
                       f"{stated:.1f}km route: too short to be the same station")
        return False
    return True


# ----------------------------------------------------------------- unit lists
def units_govtschools():
    d = json.loads((REF / "schools_source.json").read_text(encoding="utf-8"))
    G.SOURCES["govtschools"] = (
        d["source"] + " - " + "; ".join(d["source_urls"].values()),
        "Every school the Colombo Zonal Education Office lists as national, "
        "provincial or semi-government. ONE list: the original study's split "
        "into Class A (top-20 A/L performers per stream), Class B and "
        "semi-government rests on Department of Examinations ranking data that "
        "was not retrievable, so the split is not reproduced and six variables "
        "become two - see NOTES.md. Private schools are excluded (covered by "
        "intlschools.csv) and pirivenas are excluded (monastic, not general).")
    return [(r["name"], r["address"]) for r in d["rows"]], {}


def _commercial(fn, cat, label):
    d = json.loads((REF / fn).read_text(encoding="utf-8"))
    pub = {}
    units = []
    for r in d["rows"]:
        nm = f"{r['brand']} - {r['name']}"
        units.append((nm, r.get("address") or ""))
        if r.get("lat") and r.get("lon"):
            try:
                pub[nm] = (float(r["lat"]), float(r["lon"]))
            except (TypeError, ValueError):
                pass
    got = ", ".join(f"{k} {v}" for k, v in d["brands_retrieved"].items())
    dead = "; ".join(f"{k}: {v}" for k, v in d["brands_not_retrievable"].items())
    G.SOURCES[cat] = (
        d["source"],
        f"{label} Brands retrieved: {got}. NOT RETRIEVABLE, recorded as a "
        f"coverage limitation and not substituted: {dead}. De-duplicated on "
        f"(brand, address, name): {d['rows_before_dedup']} -> {len(d['rows'])} "
        f"rows before the Colombo scope filter.")
    return units, pub


def units_banks():
    return _commercial("banks_source.json", "banks",
                       "Branches of the top banking brands by brand value.")


def units_finance():
    return _commercial("finance_companies_source.json", "finance_companies",
                       "Branches of the top non-banking financial services "
                       "brands by brand value.")


def units_railway():
    rail = json.loads((REF / "railway_stations_source.json").read_text(encoding="utf-8"))
    units, km = [], {}
    for stations in rail["stations_within_max_km"].values():
        for s in stations:
            nm = f"{s['name']} Railway Station"
            units.append((nm, s["name"]))
            km[nm] = s["km_from_fort"]
    units.insert(0, ("Colombo Fort Railway Station", "Fort, Colombo"))
    km["Colombo Fort Railway Station"] = 0.0
    G.RAILWAY_KM.update(km)
    G.SOURCES["railway_stations"] = (
        "Sri Lanka Railways 'Station Details', railway.gov.lk",
        f"{rail['counts']['total']} stations in the source; the "
        f"{rail['counts']['within_max_km']} within 60 km of Fort are in scope "
        "before the Colombo scope filter, plus Colombo Fort added manually "
        "(the table measures distance FROM Fort so Fort has no row of its own). "
        "PRIMARY acceptance test is agreement with the source's published "
        f"km-from-Fort, tolerance {G.RAILWAY_KM_TOLERANCE} km - an independent "
        "measurement that does not come from a geocoder.")
    return units, {}


def units_med_centers():
    phs = json.loads((REF / "phsrc_source_lists.json").read_text(encoding="utf-8"))
    c = phs["categories"]["pvt_med_centers"]
    G.SOURCES["pvt_med_centers"] = (
        phs["source"],
        "Registrations with prefix PHSRC/MC (Medical Centres) - the register's "
        f"own named classification. De-duplicated on address: "
        f"{c['registrations']} registrations -> {c['rows_after_address_dedup']} "
        "rows before the Colombo scope filter.")
    return [(r["name"], r["address"]) for r in c["rows"]], {}


def units_fuel():
    fuel = json.loads((REF / "fuel_stations_source.json").read_text(encoding="utf-8"))
    G.SOURCES["fuel_stations"] = (
        "; ".join(fuel["sources"].get("Ceypetco", [])),
        "Ceypetco district filling-station pages for Colombo and the districts "
        "that touch it. COVERAGE LIMITATION: Lanka IOC is incomplete (24 of "
        "~202 captured; the dealer list paginates 24 per page with no controls, "
        "no select and no visible API) and Laugfs was not attempted. Neither "
        "was substituted.")
    return [(s["name"], s["raw_location"]) for s in fuel["stations"]], {}


JOBS = [
    ("govtschools", units_govtschools),
    ("finance_companies", units_finance),
    ("banks", units_banks),
    ("railway_stations", units_railway),
    ("pvt_med_centers", units_med_centers),
    ("fuel_stations", units_fuel),
]


def main() -> int:
    if not os.environ.get("GOOGLE_MAPS_API_KEY", "").strip():
        print("GOOGLE_MAPS_API_KEY is not set - this step is Google tier 1.")
        return 1

    tokens = load_scope()
    in_scope = scope_matcher(tokens)
    polys = G.build_polygon_cache()
    cache = G.load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": G.UA})
    print(f"polygons {len(polys)} | cache {len(cache)} | scope tokens {len(tokens)}")

    report, summary = [], {}
    for cat, fn in JOBS:
        units, pub = fn()
        kept, dropped = [], []
        for nm, addr in units:
            (kept if in_scope(addr) else dropped).append((nm, addr))
        print(f"\n### {cat}: {len(units)} in source, {len(kept)} in Colombo scope, "
              f"{len(dropped)} dropped before geocoding")

        G.resolve = make_resolver(pub)
        s = G.write_category(cat, kept, polys, cache, session)
        s["source_units"] = len(units)
        s["scope_dropped"] = len(dropped)
        s["published"] = sum(1 for n, _ in kept if n in pub)
        summary[cat] = s

        # audits - all three are expected to be zero
        dups, seen = [], {}
        for r in s["rows"]:
            for (a, b), owner in seen.items():
                if G.haversine(a, b, r["latitude"], r["longitude"]) * 1000 <= G.DUPLICATE_M:
                    dups.append(f"{r['name']} shares a pin with {owner}")
            seen[(r["latitude"], r["longitude"])] = r["name"]
        over = [f"{r['name']} at {r['km']}km" for r in s["rows"]
                if r["km"] is not None and r["km"] > G.ACCEPT_KM[cat]]
        outside = [r["name"] for r in s["rows"]
                   if not G.in_sri_lanka(r["latitude"], r["longitude"])]
        s["audit"] = (len(dups), len(over), len(outside))

        n, u = len(s["rows"]), len(kept)
        report.append(
            f"{cat}\n"
            f"  rows written / units in scope / coverage : "
            f"{n} / {u} / {(n / u * 100 if u else 0):.1f}%\n"
            f"  duplicates, over-radius, outside Sri Lanka : "
            f"{len(dups)} {len(over)} {len(outside)}\n"
            f"  in Colombo district : {s['in']}")
        (RESULTS / "step2n_report.txt").write_text(
            "\n".join(report), encoding="utf-8")

    print("\n" + "=" * 72)
    print("\n".join(report))
    print("=" * 72)
    for cat, s in summary.items():
        print(f"{cat:<20} source={s['source_units']:>5} dropped_pre_geocode="
              f"{s['scope_dropped']:>5} split={s['split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

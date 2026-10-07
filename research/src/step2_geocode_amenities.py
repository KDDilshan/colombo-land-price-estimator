"""Step 2 - Geocode the first four amenity categories and write their CSVs.

FOUR-TIER RESOLUTION.  Each unit is resolved by the first tier that answers,
and the tier used is recorded as `coord_source`:

  name_search  Nominatim, the official name exactly as the source register
               gives it.
  variant      Nominatim, a mechanical rewrite of that same name (drop a
               leading "The", swap "<type> <place>" to "<place> <type>",
               reduce the facility type to a generic word, append the
               locality).  Still the register's name, just re-spelled.
  overpass     The register asserts the unit exists; Overpass supplies its
               pin.  MEMBERSHIP still comes from the register - an Overpass
               result is only ever attached to a name already on the list.
  manual       Wikidata coordinate (P625) for the named unit, used where OSM
               has no representation at all.
  FAILED       Nothing found.  The unit is reported, never guessed.

A locality is NEVER queried on its own to stand in for a unit.  An earlier
version did that and returned the TOWN when the institution was missing -
putting the Open University at Karapitiya, Galle and KDU at Sooriyawewa.
Locality centroids are still computed, but only to bound the Overpass search.

Outputs, per category:
  data/amenities/<name>.csv                 name,latitude,longitude,in_colombo_district
  data/amenities/<name>_PROVENANCE.txt
  results/step2_verification.txt            every unit, coordinate, GN division

Run:  python src/step2_geocode_amenities.py
"""

from __future__ import annotations

import difflib
import json
import os
import math
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1c_geocode_unresolved import build_polygon_cache, locate  # noqa: E402
from step2_source_lists import (  # noqa: E402
    EXPRESSWAY_ENTRANCES,
    GOVT_HOSPITALS,
    INTL_SCHOOLS,
    UNIVERSITIES,
)

AMEN = ROOT / "data" / "amenities"
RESULTS = ROOT / "results"
CACHE = AMEN / "_geocode_cache.json"

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = "https://overpass-api.de/api/interpreter"
WIKIDATA = "https://www.wikidata.org/w/api.php"
UA = "colombo-land-price-replication/1.0 (MSc dissertation replication)"
SLEEP = 1.2

ALLOWED_CLASSES = {
    "place", "amenity", "building", "boundary",
    "leisure", "office", "tourism", "healthcare", "highway",
}
LK_BBOX = (5.70, 9.95, 79.40, 82.00)

OVERPASS_RADIUS_M = 3000
NAME_SIM_MIN = 0.60

# FIX 1 - acceptance radius.  A hit further than this from the unit's stated
# locality is not that unit, whichever tier produced it.  Rejected hits fall
# through to the next tier and, if nothing survives, the unit is FAILED - a
# rejected hit is never a coordinate.
ACCEPT_KM = {
    "expressway_entrances": 2.0,
    "govt_hospitals": 2.0,
    "intlschools": 2.0,
    "universities": 3.0,    # larger campuses only
}

# FIX 4 - two different units may not share a pin.  100 m apart counts as the
# same pin; distinct campuses are never that close in this data.
DUPLICATE_M = 100.0

# FIX 5 - tokens that carry no distinguishing information.  Every variant of a
# unit's name must retain all of that name's NON-generic tokens, so a variant
# can never drop "Ratnapura" or "Senior" and match a different campus.
GENERIC_TOKENS = {
    "the", "of", "and", "in", "for", "at", "sri", "lanka",
    "university", "college", "school", "schools", "hospital", "institute",
    "international", "divisional", "base", "teaching", "national", "district",
    "general", "provincial", "prison", "police", "leprosy", "campus",
    "interchange", "memorial", "centre", "center",
}

# An interchange IS a road feature, so class=highway is admissible for that
# category only, and only for junction types - never an ordinary road.
HIGHWAY_JUNCTION_TYPES = {"motorway_junction", "junction"}

# FIX 9 - DISABLED, code retained.  Nominatim/OSM carries no house-number
# geocoding for Colombo district.  Four addresses taken verbatim from the TISSL
# register returned zero hits each:
#   "189 Havelock Road, Colombo 05"      -> 0
#   "63 Elvitigala Mawatha, Colombo 08"  -> 0
#   "41 Glen Aber Place, Colombo 4"      -> 0
#   "13 Sri Mahabodhi Road, Dehiwela"    -> 0
# The original study geocoded by hand in Google Maps, which resolves these.
ADDRESS_TIER_ENABLED = False

# ---------------------------------------------------------------------------
# TIER ROUTING BY WHAT THE SOURCE PROVIDES.
# Nominatim has no house-number data for Colombo (proven: four addresses from
# the source registers, zero hits each).  For a source that carries a full
# street address, running four tiers that cannot succeed before the one that
# can is pure latency.  Only the ORDER changes; every guard still applies at
# every tier.
# ---------------------------------------------------------------------------
ADDRESS_FIRST_CATEGORIES = {"pvt_hospitals", "pvt_med_centers", "fuel_stations"}

# ---------------------------------------------------------------------------
# RAILWAY DISTANCE GUARD.
# Sri Lanka Railways publishes each station's distance from Colombo Fort.
# That is an INDEPENDENT measurement - it does not come from a geocoder - so
# for railway stations it is the primary acceptance test, applied ahead of the
# locality radius.  A station the source places 15.5 km from Fort must not
# geocode 60 km away.
# ---------------------------------------------------------------------------
FORT = (6.9333, 79.8433)
RAILWAY_KM_TOLERANCE = 3.0
RAILWAY_KM: dict[str, float] = {}

# Station-like OSM features.  Nominatim returns class=railway,type=station for
# every Sri Lankan station; OSM also uses public_transport=station.
RAIL_CLASSES = {"railway", "public_transport"}
RAIL_TYPES = {"station", "halt", "stop", "tram_stop"}

# Spelling relaxation is permitted for railway_stations ONLY.  The SLR table
# misspells names ("Mount Laviniya", "wellawatte", "Kompnnavidiya"), and the
# token-preservation rule carries the misspelling into every variant.  It is
# safe here and nowhere else because the distance-from-Fort guard backstops
# it: a wrong spelling correction lands in the wrong place and is rejected on
# distance.  Every correction applied is logged.
SPELLING_CATEGORIES = {"railway_stations"}
SPELLING_LOG: list[dict] = []

RETRIEVED = date.today().isoformat()

# Facility-type prefixes, longest first so "District General Hospital" wins
# over "General Hospital".
TYPE_PREFIXES = [
    "District General Hospital", "Provincial General Hospital",
    "Colombo East Base Hospital", "Divisional Hospital", "Teaching Hospital",
    "National Hospital", "General Hospital", "Base Hospital",
    "Prison Hospital", "Police Hospital", "Leprosy Hospital",
]

OVERPASS_TAGS = {
    "govt_hospitals": ['["amenity"="hospital"]', '["amenity"="clinic"]',
                       '["healthcare"="hospital"]'],
    "universities": ['["amenity"="university"]', '["amenity"="college"]'],
    "intlschools": ['["amenity"="school"]'],
    "expressway_entrances": ['["highway"="motorway_junction"]'],
}

SOURCES = {
    "expressway_entrances": (
        "RDA Expressway Operation Maintenance & Management Division - toll tariff "
        "schedule (Gazette Extraordinary No. 2467/49 of 2025-12-17), "
        "http://www.exway.rda.gov.lk/index.php?page=tollrates ; Seeduwa from the "
        "EOM&M homepage notice effective 2025-12-22.",
        "Every gazetted interchange on E01/E02/E03 lying in Colombo district or "
        "within 10 km of its boundary; E04 Central Expressway contributes none.",
    ),
    "govt_hospitals": (
        "Ministry of Health institution directory, "
        "https://www.health.gov.lk/health-institutions-in-sri-lanka/",
        "Every institution of every HOSPITAL type in the directory, enumerated "
        "with no district filter; Divisional Hospitals restricted to Colombo and "
        "its four adjacent districts.",
    ),
    "universities": (
        "University Grants Commission register, "
        "https://www.ugc.ac.lk/index.php?option=com_university&view=list&Itemid=25&lang=en "
        "plus the Campuses and Other Government Universities views.",
        "UGC Universities (17) + Campuses (2) + Other Government Universities (6) "
        "= 25; the 18 UGC Institutes are excluded.",
    ),
    "intlschools": (
        "TISSL member list, Wayback Machine snapshot 20220117102815 of "
        "https://tissl.lk/members.php",
        "Every school on the TISSL membership list, one row per physical campus.",
    ),
}


# --------------------------------------------------------------------------- io
def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    AMEN.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")


def in_sri_lanka(lat: float, lon: float) -> bool:
    return LK_BBOX[0] <= lat <= LK_BBOX[1] and LK_BBOX[2] <= lon <= LK_BBOX[3]


def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ---------------------------------------------------------------------------
# THE CLASS GATE.  Every geocoder response in this file becomes coordinates
# through `geocode_hits` and nowhere else.  The same defect has now appeared
# three times - Kindelpitiya (a road answered a locality question), the
# locality fallback (a town answered an institution question) and "Dehiwela"
# (a road segment answered a locality question) - so the check lives in one
# place rather than being patched per call site.
# ---------------------------------------------------------------------------
ANCHOR_CLASSES = {"place", "boundary"}


def hit_class_ok(h: dict, purpose: str, cat: str | None) -> bool:
    cls = h.get("class")
    if purpose == "anchor":
        # An anchor is a SETTLEMENT.  A road named after a town is not the town.
        return cls in ANCHOR_CLASSES
    if cls in RAIL_CLASSES:
        # A railway feature can only answer a railway-station question, and
        # only for station-like types.  Declared here rather than patched onto
        # ALLOWED_CLASSES by a runner - a runtime patch in the runner is how
        # the distance guard came to cover one code path out of four.
        return cat == "railway_stations" and h.get("type") in RAIL_TYPES
    if cls == "highway":
        # An interchange is genuinely a road feature; nothing else is.
        return cat == "expressway_entrances" and h.get("type") in HIGHWAY_JUNCTION_TYPES
    return cls in ALLOWED_CLASSES


def geocode_hits(session, q: str, purpose: str, cat: str | None = None) -> list[dict]:
    """The ONLY route from a geocoder response to usable coordinates."""
    out = []
    for h in nominatim(session, q):
        if not hit_class_ok(h, purpose, cat):
            continue
        lat, lon = float(h["lat"]), float(h["lon"])
        if not in_sri_lanka(lat, lon):
            continue
        out.append({"lat": lat, "lon": lon, "osm_class": h.get("class"),
                    "osm_type": h.get("type"),
                    "display_name": h.get("display_name", "")})
    return out


# Sinhala romanisation swaps, tried when a place name returns no settlement
# node.  "Dehiwela" matches only road segments; "Dehiwala" is the town.
SPELLING_SWAPS = [("wela", "wala"), ("wala", "wela"), ("ela", "ala"),
                  ("ala", "ela"), ("w", "v"), ("th", "t"), ("oo", "u")]


def spelling_variants(s: str) -> list[str]:
    out = [s]
    for a, b in SPELLING_SWAPS:
        if a in s.lower():
            idx = s.lower().rfind(a)
            out.append(s[:idx] + b + s[idx + len(a):])
    seen, uniq = set(), []
    for v in out:
        if v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq


# ------------------------------------------------------------------- variants
def name_variants(name: str, locality: str) -> list[str]:
    """Mechanical rewrites of the register's own name.  Never a bare locality."""
    out: list[str] = [name]

    if name.lower().startswith("the "):
        out.append(name[4:])

    for pref in TYPE_PREFIXES:
        if name.lower().startswith(pref.lower()):
            place = name[len(pref):].strip()
            if place:
                out.append(f"{place} {pref}")
                out.append(f"{pref} {place}")
                out.append(f"{place} hospital")
            break
    else:
        # "<place> <type>" already, or a non-hospital unit
        m = re.match(r"^(.*?)\s+(Hospital|University|School|College|Institute)\b(.*)$",
                     name, re.IGNORECASE)
        if m and m.group(1):
            out.append(f"{m.group(2)} {m.group(1)}".strip())

    # "University of the X" -> "X University"
    m = re.match(r"^University of (?:the )?(.+)$", name, re.IGNORECASE)
    if m:
        out.append(f"{m.group(1)} University")
        out.append(m.group(1))

    if locality:
        head = locality.split(",")[0].strip()
        base = out[1] if len(out) > 1 and name.lower().startswith("the ") else name
        if head and head.lower() not in base.lower():
            out.append(f"{base} {head}")

    # FIX 5 - a variant may drop generic words, never distinguishing ones.
    # "Lyceum International School Ratnapura" must not become
    # "School Lyceum International", which matches the Nugegoda campus.
    distinctive = {t for t in re.findall(r"[A-Za-z]+", name.lower())
                   if t not in GENERIC_TOKENS}

    seen, uniq = set(), []
    for v in out:
        v = " ".join(v.split())
        if not v or v.lower() in seen:
            continue
        toks = set(re.findall(r"[A-Za-z]+", v.lower()))
        if not distinctive.issubset(toks):
            continue
        seen.add(v.lower())
        uniq.append(v)
    return uniq or [name]


def class_ok(cat: str, h: dict) -> bool:
    """Per-category admissible OSM classes."""
    cls = h.get("class")
    if cls == "highway":
        return cat == "expressway_entrances" and h.get("type") in HIGHWAY_JUNCTION_TYPES
    return cls in ALLOWED_CLASSES


# ------------------------------------------------------------------- queries
def nominatim(session, q: str) -> list:
    for attempt in range(3):
        try:
            r = session.get(NOMINATIM, params={
                "q": q, "format": "json", "limit": 10,
                "countrycodes": "lk", "addressdetails": 1}, timeout=45)
            if r.status_code == 200:
                return r.json()
            time.sleep(5 * (attempt + 1))
        except Exception as exc:
            print(f"      ! nominatim {q[:38]!r}: {exc}")
            time.sleep(5 * (attempt + 1))
    return []


STREET_WORDS = {
    "road", "rd", "lane", "place", "mawatha", "mw", "street", "st", "avenue",
    "ave", "crescent", "gardens", "garden", "drive", "terrace", "park",
    "circus", "walk", "row",
}


def is_street_address(s: str) -> bool:
    """True for a real street address, false for a bare town name.

    FIX 9's address tier may query an address because an address IS the unit's
    location.  It may never query a town, which is not.
    """
    if not s:
        return False
    toks = re.findall(r"[A-Za-z]+", s.lower())
    has_street_word = any(t in STREET_WORDS for t in toks)
    has_house_no = bool(re.match(r"^\s*\d+[\w/\-]*\s+\w", s))
    return has_street_word or (has_house_no and "," in s)


def locality_centre(session, locality: str, cache: dict):
    """Anchor for the acceptance radius.  NEVER returned as a unit's coordinate.

    FIX 6 - tries the MOST SPECIFIC fragment first and works outwards by
    dropping leading fragments.  The previous version tried the least specific
    fragment first, which anchored "13 Sri Mahabodhi Road, Dehiwela" somewhere
    that let a Boralesgamuwa pin pass a 5 km check on a Dehiwela school.

    Returns (lat, lon, fragment_used) or None.  None means the anchor is
    UNVERIFIABLE - the caller must fail the unit, never fall back to a coarser
    centroid, because a wrong anchor makes the radius check measure distance to
    the wrong place in both directions.
    """
    if not locality:
        return None
    key = f"__ANCHOR__|{locality}"
    if key not in cache or cache[key].get("status") != "ok":
        parts = [p.strip() for p in locality.split(",") if p.strip()]
        forms, seen = [], set()
        for i in range(len(parts)):
            f = ", ".join(parts[i:])
            if f and f.lower() not in seen:
                seen.add(f.lower())
                forms.append(f)
        entry = {"status": "failed"}
        for form in forms:
            for spell in spelling_variants(form):
                hits = geocode_hits(session, f"{spell}, Sri Lanka", "anchor")
                time.sleep(SLEEP)
                if hits:
                    entry = {"status": "ok", "lat": hits[0]["lat"],
                             "lon": hits[0]["lon"], "fragment": form,
                             "spelling": spell,
                             "osm_class": hits[0]["osm_class"],
                             "display_name": hits[0]["display_name"]}
                    break
            if entry["status"] == "ok":
                break
        cache[key] = entry
        save_cache(cache)
    c = cache[key]
    if c.get("status") != "ok":
        return None
    frag = c.get("fragment", "")
    if c.get("spelling") and c["spelling"].lower() != frag.lower():
        frag = f"{frag} (as {c['spelling']})"
    return (c["lat"], c["lon"], frag)


def overpass_lookup(session, name: str, centre, cat: str, cache: dict):
    """Pin for a unit ALREADY on the source register.  Never a membership source."""
    if centre is None or cat not in OVERPASS_TAGS:
        return None
    key = f"__OVERPASS__|{cat}|{name}|{centre[0]:.4f},{centre[1]:.4f}"
    if key in cache:
        return cache[key] if cache[key].get("status") == "ok" else None

    parts = []
    for tag in OVERPASS_TAGS[cat]:
        for kind in ("node", "way", "relation"):
            parts.append(f'{kind}{tag}(around:{OVERPASS_RADIUS_M},{centre[0]},{centre[1]});')
    q = f"[out:json][timeout:60];({''.join(parts)});out center tags;"

    entry = {"status": "failed"}
    try:
        r = session.post(OVERPASS, data={"data": q}, timeout=90)
        time.sleep(2.0)
        if r.status_code == 200:
            els = r.json().get("elements", [])
            cands = []
            for e in els:
                lat = e.get("lat") or (e.get("center") or {}).get("lat")
                lon = e.get("lon") or (e.get("center") or {}).get("lon")
                if lat is None or lon is None:
                    continue
                tags = e.get("tags") or {}
                # The class gate for Overpass: the element must actually carry
                # the tag we asked for.  Overpass is queried by tag, but the
                # response is not trusted to honour it without checking.
                if not any(
                    tags.get(k) == v
                    for k, v in (("amenity", "hospital"), ("amenity", "clinic"),
                                 ("healthcare", "hospital"),
                                 ("amenity", "university"), ("amenity", "college"),
                                 ("amenity", "school"),
                                 ("highway", "motorway_junction"))
                    if f'["{k}"="{v}"]' in "".join(OVERPASS_TAGS[cat])
                ):
                    continue
                if not in_sri_lanka(lat, lon):
                    continue
                nm = tags.get("name") or ""
                cands.append({"lat": lat, "lon": lon, "osm_name": nm,
                              "sim": sim(name, nm) if nm else 0.0})
            named = [c for c in cands if c["sim"] >= NAME_SIM_MIN]
            if named:
                best = max(named, key=lambda c: c["sim"])
                ties = [c for c in named if abs(c["sim"] - best["sim"]) < 0.02]
                if len(ties) == 1:
                    entry = {"status": "ok", **best, "rule": "name similarity"}
            elif len(cands) == 1:
                entry = {"status": "ok", **cands[0], "rule": "unique amenity in radius"}
            # 2+ untied candidates -> deliberately left for the manual tier
    except Exception as exc:
        print(f"      ! overpass {name[:34]!r}: {exc}")

    cache[key] = entry
    save_cache(cache)
    return entry if entry.get("status") == "ok" else None


def wikidata_lookup(session, name: str, cache: dict, cat: str | None = None):
    """Coordinate of record (P625) for a named unit OSM does not carry."""
    key = f"__WIKIDATA__|{name}"
    if key in cache:
        return cache[key] if cache[key].get("status") == "ok" else None
    entry = {"status": "failed"}
    try:
        r = session.get(WIKIDATA, params={
            "action": "wbsearchentities", "search": name, "language": "en",
            "format": "json", "limit": 5, "type": "item"}, timeout=45)
        time.sleep(1.0)
        hits = r.json().get("search", []) if r.status_code == 200 else []
        for h in hits:
            if sim(name, h.get("label", "")) < 0.5:
                continue
            qid = h["id"]
            r2 = session.get(WIKIDATA, params={
                "action": "wbgetentities", "ids": qid, "props": "claims",
                "format": "json"}, timeout=45)
            time.sleep(1.0)
            if r2.status_code != 200:
                continue
            claims = r2.json()["entities"][qid].get("claims", {})
            if "P625" not in claims:
                continue
            # The class gate for Wikidata.  A Wikidata item has no OSM class,
            # so "instance of" (P31) stands in: an item that is a road or a
            # street cannot be an institution's location.  Road junctions stay
            # admissible for the expressway category only.
            p31 = {c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                   for c in claims.get("P31", [])
                   if c["mainsnak"].get("datavalue")}
            ROAD_ITEMS = {"Q34442", "Q79007", "Q83620", "Q537127", "Q1788454"}
            if p31 & ROAD_ITEMS and cat != "expressway_entrances":
                continue
            v = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
            lat, lon = float(v["latitude"]), float(v["longitude"])
            if in_sri_lanka(lat, lon):
                entry = {"status": "ok", "lat": lat, "lon": lon, "qid": qid,
                         "label": h.get("label", "")}
                break
    except Exception as exc:
        print(f"      ! wikidata {name[:34]!r}: {exc}")
    cache[key] = entry
    save_cache(cache)
    return entry if entry.get("status") == "ok" else None


# -------------------------------------------------------------------- resolve
GOOGLE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_ROAD_TYPES = {"route", "street_address_range", "intersection"}
GOOGLE_RESULT_TYPES_OK = {
    "establishment", "point_of_interest", "premise", "subpremise",
    "street_address", "hospital", "school", "university", "primary_school",
    "secondary_school", "health", "local_government_office", "place_of_worship",
}
GOOGLE_LOCATION_TYPES_OK = {"ROOFTOP", "RANGE_INTERPOLATED", "GEOMETRIC_CENTER"}
GOOGLE_EXPRESSWAY_EXTRA = {"intersection", "route", "point_of_interest",
                           "establishment", "premise"}


def google_accept(res: dict, cat: str) -> bool:
    """The class/quality gate, Google edition - same philosophy as hit_class_ok.

    `types` stands in for OSM class (a `route` is a road and can never locate
    an institution) and `location_type` adds a precision gate: APPROXIMATE is
    Google's own flag for a town-level centroid, which is exactly the failure
    that produced the discarded 88.7% run.
    """
    types = set(res.get("types", []))
    loc_type = res.get("geometry", {}).get("location_type", "")
    if cat == "expressway_entrances":
        return bool(types & (GOOGLE_RESULT_TYPES_OK | GOOGLE_EXPRESSWAY_EXTRA))
    if types & GOOGLE_ROAD_TYPES:
        return False
    if loc_type not in GOOGLE_LOCATION_TYPES_OK:
        return False
    return bool(types & GOOGLE_RESULT_TYPES_OK)


def google_lookup(session, name, locality, cat, cache):
    """Tier 5.  Used ONLY where OSM-based tiers cannot resolve the unit.

    Google's Maps Platform terms restrict storing and reusing geocoding
    results, whereas OSM/Nominatim is ODbL.  This dataset ships coordinate
    files with a dissertation, so Google is confined to units OSM cannot
    resolve and every such row is tagged coord_source="google".
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        return None
    ck = f"__GOOGLE__|{cat}|{name}|{locality}"
    if ck in cache:
        return cache[ck] if cache[ck].get("status") == "ok" else None

    entry = {"status": "failed"}
    for q in (f"{name}, {locality}, Sri Lanka", f"{name}, Sri Lanka"):
        try:
            r = session.get(GOOGLE_URL, params={
                "address": q, "key": key, "region": "lk",
                "components": "country:LK"}, timeout=45)
            time.sleep(0.2)
            if r.status_code != 200:
                continue
            js = r.json()
            if js.get("status") not in ("OK", "ZERO_RESULTS"):
                print(f"      ~ Google {js.get('status')} on {name[:34]!r}")
                continue
            for res in js.get("results", []):
                if not google_accept(res, cat):
                    continue
                L = res["geometry"]["location"]
                if not in_sri_lanka(L["lat"], L["lng"]):
                    continue
                entry = {"status": "ok", "lat": L["lat"], "lon": L["lng"],
                         "location_type": res["geometry"].get("location_type"),
                         "types": res.get("types", [])[:4],
                         "formatted": res.get("formatted_address", ""),
                         "query_used": q}
                break
        except Exception as exc:
            print(f"      ! google {name[:34]!r}: {exc}")
        if entry["status"] == "ok":
            break

    cache[ck] = entry
    save_cache(cache)
    return entry if entry["status"] == "ok" else None


GOOGLE_ANCHOR_TYPES_OK = {
    "locality", "sublocality", "sublocality_level_1", "neighborhood",
    "administrative_area_level_1", "administrative_area_level_2",
    "administrative_area_level_3", "administrative_area_level_4",
    "political", "postal_code",
    # A street address is a MORE precise anchor than a locality, not a less
    # precise one.  Admissible here because an anchor is only ever the origin
    # of the radius check - it is never written to a CSV as a unit's position.
    "street_address", "premise", "subpremise", "route",
}


def google_anchor(session, locality, cache):
    """Anchor from Google, for the Google tier.

    Nominatim collapses every "Colombo N" postal area onto the Colombo city
    boundary centroid, so a correct Google pin measured against a Nominatim
    anchor is rejected at 3-7 km.  Comparing like with like fixes that.
    """
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key or not locality:
        return None
    ck = f"__GANCHOR__|{locality}"
    if ck in cache:
        c = cache[ck]
        return (c["lat"], c["lon"]) if c.get("status") == "ok" else None
    # Most specific fragment first, as with the Nominatim anchor.
    parts = [p.strip() for p in locality.split(",") if p.strip()]
    forms, seen = [], set()
    for i in range(len(parts)):
        f = ", ".join(parts[i:])
        if f and f.lower() not in seen:
            seen.add(f.lower())
            forms.append(f)

    entry = {"status": "failed"}
    for form in forms:
        try:
            r = session.get(GOOGLE_URL, params={
                "address": f"{form}, Sri Lanka", "key": key, "region": "lk",
                "components": "country:LK"}, timeout=45)
            time.sleep(0.2)
            if r.status_code == 200 and r.json().get("status") == "OK":
                for res in r.json().get("results", []):
                    if not (set(res.get("types", [])) & GOOGLE_ANCHOR_TYPES_OK):
                        continue
                    L = res["geometry"]["location"]
                    if in_sri_lanka(L["lat"], L["lng"]):
                        entry = {"status": "ok", "lat": L["lat"], "lon": L["lng"],
                                 "fragment": form,
                                 "formatted": res.get("formatted_address", "")}
                        break
        except Exception as exc:
            print(f"      ! google anchor {form[:30]!r}: {exc}")
        if entry["status"] == "ok":
            break
    cache[ck] = entry
    save_cache(cache)
    return (entry["lat"], entry["lon"]) if entry["status"] == "ok" else None


def detect_anchor_collapse(cache: dict) -> dict:
    """A coordinate serving more than one distinct locality string is a collapse.

    "Colombo 4", "Colombo 5" and "Colombo 05" all resolving to
    6.9388614, 79.8542005 means the geocoder ignored the distinguishing part
    and returned a coarser administrative unit.  Costs nothing to detect and
    the anchor must not be used silently afterwards.
    """
    by_coord: dict[tuple, set] = {}
    for k, v in cache.items():
        if not k.startswith("__ANCHOR__") or v.get("status") != "ok":
            continue
        loc = k.split("|", 1)[1]
        by_coord.setdefault((round(v["lat"], 6), round(v["lon"], 6)), set()).add(loc)
    return {c: sorted(ls) for c, ls in by_coord.items() if len(ls) > 1}


def resolve(session, name, locality, cat, cache, claimed, collapsed=None) -> dict:
    """Resolve one unit.  `claimed` maps an already-taken pin to its unit."""
    key = f"V4|{cat}|{name}|{locality}"
    if key in cache and cache[key].get("status") == "ok":
        return cache[key]

    limit = ACCEPT_KM[cat]
    anchor = locality_centre(session, locality, cache)

    # FIX 2 / FIX 6 - no anchor means the acceptance radius cannot be applied,
    # so the unit is unverifiable.  Never fall back to a coarser centroid.
    if anchor is None:
        res = {"status": "failed", "coord_source": "FAILED",
               "anchor_status": "unverifiable", "anchor_fragment": "",
               "reason": f"no fragment of locality {locality!r} resolved, so the "
                         f"acceptance radius could not be applied"}
        cache[key] = res
        save_cache(cache)
        return res
    centre = (anchor[0], anchor[1])
    anchor_fragment = anchor[2]
    anchor_src = "nominatim"

    # A collapsed anchor must not be used silently.  Prefer a Google anchor,
    # which distinguishes the postal areas Nominatim merges.
    is_collapsed = bool(collapsed) and (round(anchor[0], 6), round(anchor[1], 6)) in collapsed
    if is_collapsed:
        ga = google_anchor(session, locality, cache)
        if ga:
            centre, anchor_src = ga, "google (nominatim anchor collapsed)"
        else:
            anchor_src = "nominatim (COLLAPSED, no google anchor)"

    def distinctive_of(s: str) -> set:
        return {t for t in re.findall(r"[A-Za-z]+", s.lower())
                if t not in GENERIC_TOKENS}

    my_tokens = distinctive_of(name)

    def duplicate_of(lat, lon):
        """FIX 8 - a shared pin is allowed only between related campuses."""
        for (clat, clon), owner in claimed.items():
            if owner == name:
                continue
            if haversine(clat, clon, lat, lon) * 1000 > DUPLICATE_M:
                continue
            if my_tokens & distinctive_of(owner):
                return ("shared", owner)      # e.g. Wycherley Senior / Junior
            return ("blocked", owner)          # e.g. Lyceum Ratnapura on Nugegoda
        return None

    rejects: list[str] = []

    def fort_ok(lat, lon, tag):
        """Railway primary guard: agree with the source's km-from-Fort."""
        if cat != "railway_stations":
            return True
        stated = RAILWAY_KM.get(name)
        if stated is None:
            return True
        got = haversine(FORT[0], FORT[1], lat, lon)
        if abs(got - stated) > RAILWAY_KM_TOLERANCE:
            rejects.append(f"{tag} -> {got:.1f}km from Fort, source says "
                           f"{stated:.1f}km (>{RAILWAY_KM_TOLERANCE}km apart)")
            return False
        return True

    # ---- address-first routing -------------------------------------------
    if cat in ADDRESS_FIRST_CATEGORIES and is_street_address(locality):
        gg0 = google_lookup(session, name, locality, cat, cache)
        if gg0:
            gc = google_anchor(session, locality, cache) or centre
            d0 = haversine(gc[0], gc[1], gg0["lat"], gg0["lon"])
            dup0 = duplicate_of(gg0["lat"], gg0["lon"])
            if d0 > limit:
                rejects.append(f"google(addr-first) -> {d0:.1f}km (>{limit}km)")
            elif dup0 and dup0[0] == "blocked":
                rejects.append(f"google(addr-first) -> pin claimed by {dup0[1]!r}")
            elif fort_ok(gg0["lat"], gg0["lon"], "google(addr-first)"):
                res = {"status": "ok",
                       "coord_source": "shared_campus_pin" if dup0 else "google",
                       "anchor_status": "ok", "anchor_fragment": anchor_fragment,
                       "lat": gg0["lat"], "lon": gg0["lon"],
                       "osm_class": gg0.get("location_type", ""),
                       "display_name": (f"[GOOGLE {gg0.get('location_type')}] "
                                        f"{gg0.get('formatted','')}"),
                       "km_from_locality": round(d0, 2),
                       "query_used": gg0.get("query_used", name),
                       "rejected": rejects[:6]}
                cache[key] = res
                save_cache(cache)
                return res

    # Tier order: name+locality -> address -> variants -> overpass -> wikidata
    variants = name_variants(name, locality)
    queries = [(f"{name}, {locality}, Sri Lanka", "name_search")]
    if ADDRESS_TIER_ENABLED and is_street_address(locality):
        queries.append((f"{locality}, Sri Lanka", "address"))
    queries += [(f"{v}, Sri Lanka", "name_search" if i == 0 else "variant")
                for i, v in enumerate(variants)]

    # Spelling relaxation, railway_stations only - see SPELLING_CATEGORIES.
    if cat in SPELLING_CATEGORIES:
        seen_q = {q for q, _ in queries}
        for base in (name, f"{name}, {locality}"):
            for sp in spelling_variants(base):
                if sp == base:
                    continue
                q = f"{sp}, Sri Lanka"
                if q not in seen_q:
                    seen_q.add(q)
                    queries.append((q, "variant"))

    for q, tier in queries:
        hits = geocode_hits(session, q, "result", cat)
        time.sleep(SLEEP)
        cands = []
        for h in hits:
            lat, lon = h["lat"], h["lon"]
            d = haversine(centre[0], centre[1], lat, lon)
            # The address tier locates the unit by its own street address, so
            # the radius check against the anchor is not meaningful there.
            if tier != "address" and d > limit:
                rejects.append(f"{q!r} -> {d:.1f}km (>{limit}km)")
                continue
            if not fort_ok(lat, lon, repr(q)):
                continue
            dup = duplicate_of(lat, lon)
            shared = False
            if dup:
                kind, owner = dup
                if kind == "blocked":
                    rejects.append(f"{q!r} -> pin claimed by unrelated {owner!r}")
                    continue
                shared = True
            cands.append({"lat": lat, "lon": lon, "osm_class": h["osm_class"],
                          "display_name": h["display_name"],
                          "km_from_locality": round(d, 2), "query_used": q,
                          "shared": shared})
        if cands:
            best = min(cands, key=lambda c: c["km_from_locality"])
            if (cat in SPELLING_CATEGORIES
                    and not best["query_used"].lower().startswith(name.lower())):
                stated = RAILWAY_KM.get(name)
                got = haversine(FORT[0], FORT[1], best["lat"], best["lon"])
                SPELLING_LOG.append({
                    "source_spelling": name,
                    "query_that_matched": best["query_used"],
                    "km_source": stated,
                    "km_geocoded": round(got, 2),
                    "disagreement_km": round(abs(got - stated), 2)
                    if stated is not None else None})
            src = "shared_campus_pin" if best.pop("shared", False) else tier
            for c in cands:
                c.pop("shared", None)
            res = {"status": "ok", "coord_source": src, **best,
                   "anchor_status": "ok", "anchor_fragment": anchor_fragment,
                   "rejected": rejects[:6]}
            cache[key] = res
            save_cache(cache)
            return res

    ov = overpass_lookup(session, name, centre, cat, cache)
    if ov:
        d = haversine(centre[0], centre[1], ov["lat"], ov["lon"])
        dup = duplicate_of(ov["lat"], ov["lon"])
        if d > limit:
            rejects.append(f"overpass -> {d:.1f}km (>{limit}km)")
        elif dup and dup[0] == "blocked":
            rejects.append(f"overpass -> pin claimed by unrelated {dup[1]!r}")
        elif not fort_ok(ov["lat"], ov["lon"], "overpass"):
            pass
        else:
            res = {"status": "ok",
                   "coord_source": "shared_campus_pin" if dup else "overpass",
                   "anchor_status": "ok", "anchor_fragment": anchor_fragment,
                   "lat": ov["lat"],
                   "lon": ov["lon"], "osm_class": "overpass",
                   "display_name": f"OSM name={ov.get('osm_name','')} [{ov.get('rule')}]",
                   "km_from_locality": round(d, 2), "query_used": name,
                   "rejected": rejects[:6]}
            cache[key] = res
            save_cache(cache)
            return res

    wd = wikidata_lookup(session, name, cache, cat)
    if wd:
        d = haversine(centre[0], centre[1], wd["lat"], wd["lon"])
        dup = duplicate_of(wd["lat"], wd["lon"])
        if d > limit:
            rejects.append(f"wikidata -> {d:.1f}km (>{limit}km)")
        elif dup and dup[0] == "blocked":
            rejects.append(f"wikidata -> pin claimed by unrelated {dup[1]!r}")
        elif not fort_ok(wd["lat"], wd["lon"], "wikidata"):
            pass
        else:
            res = {"status": "ok",
                   "coord_source": "shared_campus_pin" if dup else "wikidata",
                   "anchor_status": "ok", "anchor_fragment": anchor_fragment,
                   "lat": wd["lat"],
                   "lon": wd["lon"], "osm_class": "wikidata",
                   "display_name": f"Wikidata {wd['qid']} ({wd.get('label','')})",
                   "km_from_locality": round(d, 2), "query_used": name,
                   "rejected": rejects[:6]}
            cache[key] = res
            save_cache(cache)
            return res

    # Tier 5 - Google, only for what OSM could not resolve.
    gg = google_lookup(session, name, locality, cat, cache)
    if gg:
        # Measure a Google pin against a Google anchor - like with like.
        gcentre = google_anchor(session, locality, cache) or centre
        d = haversine(gcentre[0], gcentre[1], gg["lat"], gg["lon"])
        dup = duplicate_of(gg["lat"], gg["lon"])
        if d > limit:
            rejects.append(f"google -> {d:.1f}km (>{limit}km)")
        elif dup and dup[0] == "blocked":
            rejects.append(f"google -> pin claimed by unrelated {dup[1]!r}")
        elif not fort_ok(gg["lat"], gg["lon"], "google"):
            pass
        else:
            res = {"status": "ok",
                   "coord_source": "shared_campus_pin" if dup else "google",
                   "anchor_status": "ok", "anchor_fragment": anchor_fragment,
                   "lat": gg["lat"], "lon": gg["lon"],
                   "osm_class": gg.get("location_type", ""),
                   "display_name": (f"[GOOGLE {gg.get('location_type')}] "
                                    f"{gg.get('formatted','')} "
                                    f"types={','.join(gg.get('types', []))}"),
                   "km_from_locality": round(d, 2),
                   "query_used": gg.get("query_used", name),
                   "rejected": rejects[:6]}
            cache[key] = res
            save_cache(cache)
            return res

    res = {"status": "failed", "coord_source": "FAILED",
           "anchor_status": "ok", "anchor_fragment": anchor_fragment,
           "anchor_lat": centre[0], "anchor_lon": centre[1],
           "tried": [q for q, _ in queries], "rejected": rejects[:10]}
    cache[key] = res
    save_cache(cache)
    return res


# --------------------------------------------------------------------- output
def load_manual() -> dict:
    """Tier 6 - hand-determined coordinates, each with a verifiable reference.

    Only units that every automated tier failed appear here.  A bare lat/long
    is not acceptable: every row records how the point was determined and what
    corroborates it (a published address, or the feature name at that point).
    """
    path = AMEN / "_manual_coordinates.csv"
    if not path.exists():
        return {}
    import csv
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("name"):
                out[r["name"].strip()] = {
                    "lat": float(r["latitude"]), "lon": float(r["longitude"]),
                    "how": r.get("determined_how", ""),
                    "ref": r.get("verifiable_reference", "")}
    return out


MANUAL = load_manual()


def write_category(cat, units, polys, cache, session, type_labels=None,
                   collapsed=None):
    print(f"\n=== {cat}: {len(units)} units (accept radius {ACCEPT_KM[cat]}km) ===")
    rows, failures = [], []
    claimed: dict[tuple, str] = {}
    for i, (name, locality) in enumerate(units, 1):
        r = resolve(session, name, locality, cat, cache, claimed, collapsed)
        if r["status"] != "ok" and name in MANUAL:
            m = MANUAL[name]
            r = {"status": "ok", "coord_source": "manual",
                 "lat": m["lat"], "lon": m["lon"], "osm_class": "manual",
                 "display_name": f"[MANUAL] {m['how'][:70]} || REF: {m['ref'][:70]}",
                 "km_from_locality": None,
                 "anchor_fragment": r.get("anchor_fragment", ""),
                 "query_used": "hand-determined, see _manual_coordinates.csv"}
        if r["status"] != "ok":
            near = ""
            if r.get("anchor_lat") is not None:
                ap = locate(r["anchor_lon"], r["anchor_lat"], polys)
                if ap:
                    near = "anchor IN Colombo district"
                else:
                    dmin = min((haversine(r["anchor_lat"], r["anchor_lon"], y, x)
                                for poly in polys for ring in poly["rings"]
                                for x, y in ring[::7]), default=999.0)
                    near = (f"anchor {dmin:.1f}km from Colombo boundary"
                            + (" (INSIDE 10km buffer)" if dmin <= 10 else ""))
            failures.append((name, locality,
                             (r.get("reason", "") + " | " + near).strip(" |")))
            print(f"  [{i:3d}/{len(units)}] FAILED  {name}")
            continue
        claimed[(r["lat"], r["lon"])] = name
        poly = locate(r["lon"], r["lat"], polys)
        rows.append({"name": name, "locality": locality,
                     "latitude": round(r["lat"], 7), "longitude": round(r["lon"], 7),
                     "in_colombo_district": 1 if poly else 0,
                     "coord_source": r["coord_source"],
                     "gn": poly["base_name"] if poly else "",
                     "ds": poly["ds"] if poly else "",
                     "km": r.get("km_from_locality"),
                     "anchor": r.get("anchor_fragment", ""),
                     "display": r.get("display_name", ""),
                     "query": r.get("query_used", "")})
        if i % 25 == 0:
            print(f"  [{i:3d}/{len(units)}] ...")

    AMEN.mkdir(parents=True, exist_ok=True)
    with open(AMEN / f"{cat}.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write("name,latitude,longitude,in_colombo_district\n")
        for r in rows:
            nm = r["name"].replace('"', "'")
            nm = f'"{nm}"' if "," in nm else nm
            fh.write(f"{nm},{r['latitude']},{r['longitude']},{r['in_colombo_district']}\n")

    split: dict[str, int] = {}
    for r in rows:
        split[r["coord_source"]] = split.get(r["coord_source"], 0) + 1
    split["FAILED"] = len(failures)
    n_in = sum(r["in_colombo_district"] for r in rows)

    src, rule = SOURCES[cat]
    p = ["=" * 78, f"PROVENANCE - {cat}.csv", "=" * 78,
         f"Source URL      : {src}", f"Date retrieved  : {RETRIEVED}",
         f"Selection rule  : {rule}", "",
         f"Units in source list : {len(units)}", f"Rows written         : {len(rows)}",
         f"  in_colombo_district = 1 : {n_in}",
         f"  in_colombo_district = 0 : {len(rows) - n_in}", "",
         "coord_source split:"]
    for k in ("name_search", "address", "variant", "overpass", "wikidata", "google", "shared_campus_pin", "manual", "FAILED"):
        p.append(f"  {k:<12} {split.get(k, 0):>4}")
    p += ["",
          "name_search = Nominatim on the register's official name.",
          "variant     = Nominatim on a mechanical rewrite of that same name.",
          "overpass    = the register asserts the unit exists; Overpass supplied",
          f"              the pin, bounded to {OVERPASS_RADIUS_M/1000:.0f} km of the stated locality and",
          f"              requiring name similarity >= {NAME_SIM_MIN} or a unique amenity",
          "              of the right type. Membership is still the register's.",
          "manual      = Wikidata coordinate (P625) where OSM has no feature.",
          "", ]
    if failures:
        p.append("UNITS THAT COULD NOT BE GEOCODED:")
        for n, l, why in failures:
            p.append(f"  - {n}  ({l})" + (f"\n      reason: {why}" if why else ""))
        p.append("")
    if type_labels:
        p.append("DIRECTORY TYPE LABELS (the source register's own classification):")
        for t, names in type_labels.items():
            p.append(f"  {t}: {len(names)}")
        p.append("")
    p.append("EVERY UNIT - coordinate, GN division, coord_source:")
    for r in sorted(rows, key=lambda x: (-x["in_colombo_district"], x["name"])):
        p.append(f"  [{r['in_colombo_district']}] {r['name']}")
        p.append(f"        {r['latitude']}, {r['longitude']}  src={r['coord_source']}"
                 f"  {r['km']}km from locality  GN={r['gn']}  DS={r['ds']}")
        p.append(f"        query={r['query']!r}")
        p.append(f"        {r['display'][:110]}")
    (AMEN / f"{cat}_PROVENANCE.txt").write_text("\n".join(p), encoding="utf-8")

    print(f"  wrote {cat}.csv: {len(rows)} rows, {n_in} in Colombo, "
          f"{len(failures)} failed | {split}")
    return {"rows": rows, "failures": failures, "split": split,
            "units": len(units), "in": n_in}


def main() -> int:
    polys = build_polygon_cache()
    cache = load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    print(f"Colombo ADM4 polygons: {len(polys)} | cache entries: {len(cache)}")

    # ---- anchor pre-pass, so collapses are known before any unit resolves --
    all_units = list(EXPRESSWAY_ENTRANCES) + list(INTL_SCHOOLS)
    for lst in GOVT_HOSPITALS.values():
        all_units += lst
    for lst in UNIVERSITIES.values():
        all_units += lst
    print(f"\nanchor pre-pass over {len(all_units)} units ...")
    for n, (nm, loc) in enumerate(all_units, 1):
        locality_centre(session, loc, cache)
        if n % 50 == 0:
            print(f"  anchors {n}/{len(all_units)}")
    collapsed = detect_anchor_collapse(cache)
    print(f"anchor collapses detected: {len(collapsed)}")
    log = ["=" * 78, "ANCHOR COLLAPSE LOG", "=" * 78,
           "A coordinate serving more than one distinct locality string means the",
           "geocoder ignored the distinguishing part and returned a coarser unit.",
           ""]
    for (lat, lon), locs in sorted(collapsed.items(), key=lambda kv: -len(kv[1])):
        log.append(f"{lat}, {lon}  <- {len(locs)} distinct locality strings:")
        for l in locs:
            log.append(f"      {l}")
        log.append("")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step2_anchor_collapse.txt").write_text("\n".join(log), encoding="utf-8")

    summary = {}
    summary["expressway_entrances"] = write_category(
        "expressway_entrances", EXPRESSWAY_ENTRANCES, polys, cache, session,
        collapsed=collapsed)

    hu, ht = [], {}
    for label, lst in GOVT_HOSPITALS.items():
        ht[label] = lst
        hu.extend(lst)
    summary["govt_hospitals"] = write_category(
        "govt_hospitals", hu, polys, cache, session, ht, collapsed=collapsed)

    uu, ut = [], {}
    for label, lst in UNIVERSITIES.items():
        ut[label] = lst
        uu.extend(lst)
    summary["universities"] = write_category(
        "universities", uu, polys, cache, session, ut, collapsed=collapsed)

    summary["intlschools"] = write_category(
        "intlschools", INTL_SCHOOLS, polys, cache, session, collapsed=collapsed)

    # ---- verification listing -------------------------------------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    v = ["=" * 78, "STEP 2 VERIFICATION - every unit, coordinate, GN division",
         "=" * 78, ""]
    for cat, s in summary.items():
        v.append("=" * 78)
        v.append(f"{cat}  ({len(s['rows'])} rows, {s['in']} in Colombo district)")
        v.append("=" * 78)
        for r in sorted(s["rows"], key=lambda x: (-x["in_colombo_district"], x["name"])):
            flag = "IN " if r["in_colombo_district"] else "out"
            v.append(f"{flag} {r['latitude']:>10.6f},{r['longitude']:>11.6f}  "
                     f"{str(r['km']) + 'km':>9}  {r['coord_source']:<18} "
                     f"{r['gn'][:20]:<22} {r['name']}")
        for n, l, why in s["failures"]:
            v.append(f"--- {'':>10} {'':>11}  {'':>9}  {'FAILED':<11} "
                     f"{'':<22} {n}")
        # audit: duplicate pins and over-radius rows must both be zero
        dups, seen = [], {}
        for r in s["rows"]:
            for (clat, clon), owner in seen.items():
                if haversine(clat, clon, r["latitude"], r["longitude"]) * 1000 <= DUPLICATE_M:
                    dups.append(f"{r['name']} shares a pin with {owner}")
            seen[(r["latitude"], r["longitude"])] = r["name"]
        over = [f"{r['name']} at {r['km']}km" for r in s["rows"]
                if r["km"] is not None and r["km"] > ACCEPT_KM[cat]]
        v.append("")
        v.append(f"  duplicate pins remaining : {len(dups)}")
        for d in dups:
            v.append(f"    ! {d}")
        v.append(f"  rows over the {ACCEPT_KM[cat]}km radius : {len(over)}")
        for o in over:
            v.append(f"    ! {o}")
        v.append("")
    (RESULTS / "step2_verification.txt").write_text("\n".join(v), encoding="utf-8")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    tot = {"units": 0, "rows": 0}
    agg: dict[str, int] = {}
    for cat, s in summary.items():
        n = len(s["rows"])
        print(f"{cat:<24} units={s['units']:>4} rows={n:>4} in_colombo={s['in']:>4} "
              f"failed={len(s['failures']):>3} success={n/s['units']*100:5.1f}%")
        print(f"{'':24}   split={s['split']}")
        tot["units"] += s["units"]; tot["rows"] += n
        for k, c in s["split"].items():
            agg[k] = agg.get(k, 0) + c
    print(f"{'TOTAL':<24} units={tot['units']:>4} rows={tot['rows']:>4} "
          f"success={tot['rows']/tot['units']*100:5.1f}%")
    print(f"coord_source split: {agg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

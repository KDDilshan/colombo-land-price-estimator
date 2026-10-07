"""Step 2c - Geocoder comparison: Google Maps vs Nominatim on the same 46 units.

Runs the 46 near-Colombo units that Nominatim could not resolve, plus the four
street addresses that returned zero hits, through the Google Geocoding API.

THE GUARDS ARE THE SAME.  Google is a better geocoder, not a reason to drop
them.  Each Nominatim guard has a Google equivalent:

  OSM class filter    ->  Google `types` filter.  A result typed `route` is a
                          road and can never answer "where is this place",
                          exactly as class=highway could not.
  (new) quality gate  ->  Google `location_type`.  APPROXIMATE is a town-level
                          centroid - the precise failure mode that produced the
                          fake 88.7% run - so it is rejected for unit results
                          and allowed only for anchors.
  acceptance radius   ->  unchanged: 2 km, 3 km for universities.
  duplicate pin       ->  unchanged: 100 m, shared only between units sharing a
                          distinctive token.
  Sri Lanka bbox      ->  unchanged.

Requires GOOGLE_MAPS_API_KEY in the environment.  The key is read from the
environment only - never hard-coded, never logged.

Output: results/step2c_google_comparison.txt

Run:  set GOOGLE_MAPS_API_KEY=...   then
      python src/step2c_google_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1c_geocode_unresolved import build_polygon_cache, locate  # noqa: E402
from step2_geocode_amenities import (  # noqa: E402
    ACCEPT_KM, DUPLICATE_M, GENERIC_TOKENS, LK_BBOX, haversine, in_sri_lanka,
)

AMEN = ROOT / "data" / "amenities"
RESULTS = ROOT / "results"
GOOGLE = "https://maps.googleapis.com/maps/api/geocode/json"
SLEEP = 0.15  # Google allows far higher throughput than Nominatim

# Google `types` that are a road, not a place.  Direct analogue of class=highway.
ROAD_TYPES = {"route", "street_address_range", "intersection"}
# Acceptable for a unit: an actual establishment or building, not an area.
RESULT_TYPES_OK = {
    "establishment", "point_of_interest", "premise", "subpremise",
    "street_address", "hospital", "school", "university", "primary_school",
    "secondary_school", "health", "local_government_office", "place_of_worship",
}
ANCHOR_TYPES_OK = {
    "locality", "sublocality", "sublocality_level_1", "neighborhood",
    "administrative_area_level_1", "administrative_area_level_2",
    "administrative_area_level_3", "administrative_area_level_4", "political",
    "postal_code",
}
# Google's own precision flag.  APPROXIMATE == town centroid.
RESULT_LOCATION_TYPES_OK = {"ROOFTOP", "RANGE_INTERPOLATED", "GEOMETRIC_CENTER"}

# Interchanges are road features; allow the road-ish types for that category.
EXPRESSWAY_EXTRA_TYPES = {"intersection", "route", "point_of_interest",
                          "establishment", "premise"}

ADDRESS_PROBE = [
    "189 Havelock Road, Colombo 05, Sri Lanka",
    "63 Elvitigala Mawatha, Colombo 08, Sri Lanka",
    "41 Glen Aber Place, Colombo 4, Sri Lanka",
    "13 Sri Mahabodhi Road, Dehiwela, Sri Lanka",
]


def distinctive(s: str) -> set:
    import re
    return {t for t in re.findall(r"[A-Za-z]+", s.lower()) if t not in GENERIC_TOKENS}


def google(session, q: str, key: str) -> list:
    for attempt in range(3):
        try:
            r = session.get(GOOGLE, params={"address": q, "key": key,
                                            "region": "lk",
                                            "components": "country:LK"}, timeout=45)
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            js = r.json()
            st = js.get("status")
            if st == "OK":
                return js.get("results", [])
            if st == "ZERO_RESULTS":
                return []
            print(f"      ~ Google status={st} {js.get('error_message','')[:80]}")
            if st in ("OVER_QUERY_LIMIT", "UNKNOWN_ERROR"):
                time.sleep(3 * (attempt + 1))
                continue
            return []
        except Exception as exc:
            print(f"      ! {q[:40]!r}: {exc}")
            time.sleep(2 * (attempt + 1))
    return []


def accept(res: dict, purpose: str, cat: str | None) -> bool:
    """The class/quality gate, Google edition."""
    types = set(res.get("types", []))
    loc_type = res.get("geometry", {}).get("location_type", "")
    if purpose == "anchor":
        return bool(types & ANCHOR_TYPES_OK)
    if cat == "expressway_entrances":
        return bool(types & (RESULT_TYPES_OK | EXPRESSWAY_EXTRA_TYPES))
    if types & ROAD_TYPES:
        return False
    if loc_type not in RESULT_LOCATION_TYPES_OK:
        return False
    return bool(types & RESULT_TYPES_OK)


def main() -> int:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        print("GOOGLE_MAPS_API_KEY is not set in the environment.")
        print("This script cannot run without it, and the key must come from you -")
        print("I cannot create a Google Cloud account or obtain an API key.")
        return 2

    units_path = AMEN / "_near_colombo_failures.json"
    if not units_path.exists():
        print("Run step2b_extract_near_colombo_failures.py first.")
        return 1
    data = json.loads(units_path.read_text(encoding="utf-8"))

    polys = build_polygon_cache()
    session = requests.Session()
    out: list[str] = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("=" * 78)
    emit("STEP 2c - GOOGLE MAPS vs NOMINATIM on the same 46 near-Colombo failures")
    emit("=" * 78)
    emit("Same guards as Nominatim: types filter (road rejected), location_type")
    emit("quality gate (APPROXIMATE rejected), acceptance radius, duplicate-pin")
    emit("check with the shared-token rule, Sri Lanka bounding box.")
    emit("")

    emit("--- probe: the four addresses Nominatim returned ZERO hits for ---")
    for q in ADDRESS_PROBE:
        res = google(session, q, key)
        time.sleep(SLEEP)
        if not res:
            emit(f"  0 hits   {q}")
        else:
            r0 = res[0]
            loc = r0["geometry"]["location"]
            emit(f"  {len(res)} hits   {q}")
            emit(f"           -> {loc['lat']:.6f},{loc['lng']:.6f}  "
                 f"{r0['geometry'].get('location_type')}  "
                 f"types={','.join(r0.get('types', [])[:3])}")
            emit(f"              {r0.get('formatted_address','')[:74]}")
    emit("")

    totals = {"units": 0, "resolved": 0}
    for cat, units in data.items():
        if not units:
            continue
        limit = ACCEPT_KM[cat]
        emit("=" * 78)
        emit(f"{cat}  ({len(units)} units, accept radius {limit}km)")
        emit("=" * 78)
        claimed: dict[tuple, str] = {}
        n_ok = 0
        for u in units:
            name, locality = u["name"], u["locality"]
            totals["units"] += 1

            anchor = None
            for res in google(session, f"{locality}, Sri Lanka", key):
                if accept(res, "anchor", None):
                    L = res["geometry"]["location"]
                    if in_sri_lanka(L["lat"], L["lng"]):
                        anchor = (L["lat"], L["lng"])
                        break
            time.sleep(SLEEP)

            picked, why = None, ""
            for q in (f"{name}, {locality}, Sri Lanka", f"{name}, Sri Lanka"):
                for res in google(session, q, key):
                    if not accept(res, "result", cat):
                        continue
                    L = res["geometry"]["location"]
                    lat, lon = L["lat"], L["lng"]
                    if not in_sri_lanka(lat, lon):
                        continue
                    if anchor:
                        d = haversine(anchor[0], anchor[1], lat, lon)
                        if d > limit:
                            why = f"rejected {d:.1f}km > {limit}km"
                            continue
                    else:
                        d = None
                    dup = None
                    for (clat, clon), owner in claimed.items():
                        if owner != name and haversine(clat, clon, lat, lon) * 1000 <= DUPLICATE_M:
                            dup = owner
                            break
                    if dup and not (distinctive(name) & distinctive(dup)):
                        why = f"rejected: pin claimed by unrelated {dup!r}"
                        continue
                    picked = {"lat": lat, "lon": lon, "d": d,
                              "loc_type": res["geometry"].get("location_type"),
                              "types": res.get("types", []),
                              "addr": res.get("formatted_address", ""),
                              "shared": bool(dup)}
                    break
                time.sleep(SLEEP)
                if picked:
                    break

            if picked:
                claimed[(picked["lat"], picked["lon"])] = name
                poly = locate(picked["lon"], picked["lat"], polys)
                gn = poly["base_name"] if poly else ""
                flag = "IN " if poly else "out"
                dtxt = f"{picked['d']:.2f}km" if picked["d"] is not None else "no-anchor"
                emit(f"{flag} {picked['lat']:>10.6f},{picked['lon']:>11.6f}  {dtxt:>9}  "
                     f"{picked['loc_type']:<18} {gn[:20]:<22} {name}")
                emit(f"        {picked['addr'][:96]}")
                n_ok += 1
                totals["resolved"] += 1
            else:
                emit(f"--- {'':>10} {'':>11}  {'':>9}  {'FAILED':<18} {'':<22} {name}")
                if why:
                    emit(f"        {why}")
        emit("")
        emit(f"  resolved {n_ok} of {len(units)}")
        emit("")

    emit("=" * 78)
    emit(f"GOOGLE RESOLVED {totals['resolved']} of {totals['units']} "
         f"near-Colombo units that Nominatim could not "
         f"({totals['resolved']/max(totals['units'],1)*100:.1f}%)")
    emit("Decision threshold agreed with the user: >=30 of 46 -> switch all")
    emit("fourteen categories to Google; <30 -> manual coordinates instead.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step2c_google_comparison.txt").write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

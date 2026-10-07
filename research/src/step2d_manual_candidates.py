"""Step 2d - Generate reviewable candidates for the manual coordinates.

This does NOT write coordinates.  It gathers evidence for the units that every
automated tier failed, so each one can be judged by hand and then recorded with
a verifiable reference.  A bare lat/long is not an acceptable manual
coordinate; the point must be corroborated by a published address or by the
feature name visible at that location.

For each unit it tries several query forms and prints, for every candidate:
the coordinate, Google's formatted_address, the location_type and the distance
from the register's stated locality.

Output: results/step2d_manual_candidates.txt

Run:  python src/step2d_manual_candidates.py
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
from step2_geocode_amenities import haversine, in_sri_lanka  # noqa: E402

AMEN = ROOT / "data" / "amenities"
RESULTS = ROOT / "results"
GOOGLE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Query forms per unit.  Where a published address was found by research it is
# listed first and cited in the comment.
CANDIDATE_QUERIES = {
    "Prison Hospital Welikade": [
        "Prison Hospital, Welikada Prison, Baseline Road, Borella, Colombo",
        "Welikada Prison, Borella, Colombo",
        "Welikada Prison Hospital, Colombo",
    ],
    "Wijaya Kumaranatunga Memorial Hospital": [
        # Published: Negombo-Colombo Main Rd, Seeduwa (wijayakmhospital.com)
        "Wijaya Kumaratunga Memorial Hospital, Negombo-Colombo Main Road, Seeduwa",
        "Wijaya Kumaratunga Memorial Hospital, Seeduwa",
    ],
    "Sri Jayawardenapura General Hospital": [
        # Published: Thalapathpitiya, Nugegoda 10250 (sjgh.health.gov.lk)
        "Sri Jayewardenepura General Hospital, Thalapathpitiya, Nugegoda",
        "Sri Jayewardenepura General Hospital, Kotte",
    ],
    "Neville Fernando Hospital Malabe": [
        "Neville Fernando Teaching Hospital, Malabe",
        "Neville Fernando Hospital, Chandrika Kumaratunga Mawatha, Malabe",
    ],
    "Colombo East Base Hospital Mulleriyawa": [
        "Colombo East Base Hospital, Mulleriyawa New Town",
        "Base Hospital Mulleriyawa, Mulleriyawa",
        "Mulleriyawa Base Hospital, Angoda",
    ],
    "Divisional Hospital Athurugiriya": [
        "Divisional Hospital Athurugiriya, Athurugiriya",
        "Athurugiriya Government Hospital, Athurugiriya",
        "District Hospital Athurugiriya",
    ],
    "Divisional Hospital Colombo Central": [
        "Divisional Hospital Colombo Central, Maligawatta, Colombo 10",
        "Colombo Central Hospital, Maligawatta",
        "Maligawatta Government Hospital, Colombo",
    ],
    "Divisional Hospital Moratuwa": [
        "Divisional Hospital Moratuwa, Moratuwa",
        "Moratuwa Government Hospital, Moratuwa",
        "District Hospital Moratuwa",
    ],
    "Divisional Hospital Hinguralakanda": [
        "Divisional Hospital Hinguralakanda, Kegalle",
        "Hinguralakanda Hospital, Kegalle",
    ],
    "Divisional Hospital Gonagaldeniya": [
        "Divisional Hospital Gonagaldeniya, Ruwanwella, Kegalle",
        "Gonagaldeniya Hospital, Ruwanwella",
    ],
    "Kerawalapitiya Interchange KIC2": [
        "Kerawalapitiya Interchange, Outer Circular Expressway, Kerawalapitiya",
        "Kerawalapitiya Interchange, Wattala",
    ],
}


def main() -> int:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        print("GOOGLE_MAPS_API_KEY not set.")
        return 2

    units = json.loads((AMEN / "_near_colombo_failures.json").read_text(encoding="utf-8"))
    flat = {}
    for cat, us in units.items():
        for u in us:
            flat[u["name"]] = (cat, u["locality"], u["reason"])

    cache = json.loads((AMEN / "_geocode_cache.json").read_text(encoding="utf-8"))
    polys = build_polygon_cache()
    session = requests.Session()
    out: list[str] = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("=" * 78)
    emit("STEP 2d - CANDIDATES FOR MANUAL COORDINATES (review only, nothing written)")
    emit("=" * 78)
    emit("")

    for name, queries in CANDIDATE_QUERIES.items():
        cat, locality, reason = flat.get(name, ("?", "?", ""))
        anchor = None
        ak = f"__ANCHOR__|{locality}"
        if ak in cache and cache[ak].get("status") == "ok":
            anchor = (cache[ak]["lat"], cache[ak]["lon"])
        emit("=" * 78)
        emit(f"{name}")
        emit(f"  category  : {cat}")
        emit(f"  register  : locality={locality!r}")
        emit(f"  why failed: {reason}")
        emit("")
        for q in queries:
            try:
                r = session.get(GOOGLE_URL, params={
                    "address": f"{q}, Sri Lanka", "key": key, "region": "lk",
                    "components": "country:LK"}, timeout=45)
                time.sleep(0.2)
                js = r.json() if r.status_code == 200 else {}
                res = js.get("results", []) if js.get("status") == "OK" else []
            except Exception as exc:
                emit(f"  ! {q[:60]}: {exc}")
                continue
            if not res:
                emit(f"  0 hits  {q}")
                continue
            emit(f"  {len(res)} hits  {q}")
            for h in res[:2]:
                L = h["geometry"]["location"]
                if not in_sri_lanka(L["lat"], L["lng"]):
                    continue
                d = (haversine(anchor[0], anchor[1], L["lat"], L["lng"])
                     if anchor else None)
                poly = locate(L["lng"], L["lat"], polys)
                emit(f"      {L['lat']:.6f}, {L['lng']:.6f}  "
                     f"{h['geometry'].get('location_type'):<18} "
                     f"{'IN ' if poly else 'out'} {(poly['base_name'] if poly else ''):<20} "
                     f"{('%.2fkm' % d) if d is not None else 'no-anchor'}")
                emit(f"        types={','.join(h.get('types', [])[:4])}")
                emit(f"        {h.get('formatted_address','')[:90]}")
        emit("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step2d_manual_candidates.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'step2d_manual_candidates.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

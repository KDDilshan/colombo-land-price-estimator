"""Step 2i - Geocode railway_stations and pvt_hospitals.

railway_stations
  Source : Sri Lanka Railways "Station Details" (step2g).
  Scope  : 263 stations in the source; the 104 within 60 km of Colombo Fort are
           in scope.  Northern, Matale and Trincomalee lines contribute none -
           no station on them can lie within 10 km of the Colombo boundary.
  Guard  : PRIMARY acceptance test is agreement with the source's published
           distance from Fort, tolerance 3 km.  Independent of any geocoder,
           so it is applied ahead of the locality radius.

pvt_hospitals
  Source : PHSRC MIS register (step2e), de-duplicated on address (step2f).
  Routing: address-first, because the register carries full street addresses
           and Nominatim has no house-number data for Colombo.

Run:  python src/step2i_run_railway_and_pvthospitals.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import step2_geocode_amenities as G  # noqa: E402

REF = ROOT / "data" / "reference"


def build_units():
    rail = json.loads((REF / "railway_stations_source.json").read_text(encoding="utf-8"))
    rail_units, rail_km = [], {}
    for line, stations in rail["stations_within_max_km"].items():
        for s in stations:
            nm = f"{s['name']} Railway Station"
            rail_units.append((nm, s["name"]))
            rail_km[nm] = s["km_from_fort"]

    phs = json.loads((REF / "phsrc_source_lists.json").read_text(encoding="utf-8"))
    ph_rows = phs["categories"]["pvt_hospitals"]["rows"]
    ph_units = [(r["name"], r["address"]) for r in ph_rows]
    return rail_units, rail_km, ph_units, rail, phs


def main() -> int:
    rail_units, rail_km, ph_units, rail, phs = build_units()
    G.RAILWAY_KM.update(rail_km)

    G.ACCEPT_KM["railway_stations"] = 2.0
    G.ACCEPT_KM["pvt_hospitals"] = 2.0
    G.OVERPASS_TAGS["railway_stations"] = ['["railway"="station"]', '["railway"="halt"]']
    G.OVERPASS_TAGS["pvt_hospitals"] = ['["amenity"="hospital"]', '["amenity"="clinic"]']
    G.ALLOWED_CLASSES = G.ALLOWED_CLASSES | {"railway"}
    G.SOURCES["railway_stations"] = (
        "Sri Lanka Railways, 'Station Details', "
        "railway.gov.lk/web/index.php?option=com_content&view=article&id=165&Itemid=191",
        f"All {rail['counts']['total']} stations in the source; the "
        f"{rail['counts']['within_max_km']} within 60 km of Colombo Fort are in "
        "scope. Primary acceptance test is agreement with the source's published "
        "distance from Fort (tolerance 3 km), applied ahead of the locality radius "
        "because it is an independent measurement.")
    G.SOURCES["pvt_hospitals"] = (
        phs["source"] + "  (partial listing also at phsrc.lk/pages_e.php?id=12)",
        "Every registration with prefix PHSRC/PH (Private Hospitals, Nursing "
        "Homes & Maternity Homes) - the register's own named classification, not "
        "a residual bucket. De-duplicated on ADDRESS, because two registrations "
        "in one building are one hospital for both the distance and the count "
        "variable: "
        f"{phs['categories']['pvt_hospitals']['registrations']} registrations -> "
        f"{phs['categories']['pvt_hospitals']['rows_after_address_dedup']} rows.")

    polys = G.build_polygon_cache()
    cache = G.load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": G.UA})
    print(f"polygons {len(polys)} | cache {len(cache)} | "
          f"google_key={'yes' if os.environ.get('GOOGLE_MAPS_API_KEY') else 'NO'}")

    all_units = rail_units + ph_units
    print(f"\nanchor pre-pass over {len(all_units)} units ...")
    for i, (nm, loc) in enumerate(all_units, 1):
        G.locality_centre(session, loc, cache)
        if i % 40 == 0:
            print(f"  anchors {i}/{len(all_units)}")
    collapsed = G.detect_anchor_collapse(cache)
    print(f"anchor collapses: {len(collapsed)}")

    summary = {}
    summary["railway_stations"] = G.write_category(
        "railway_stations", rail_units, polys, cache, session, collapsed=collapsed)
    summary["pvt_hospitals"] = G.write_category(
        "pvt_hospitals", ph_units, polys, cache, session, collapsed=collapsed)

    # ---- railway distance agreement among ACCEPTED rows ------------------
    rows = summary["railway_stations"]["rows"]
    devs = []
    for r in rows:
        stated = rail_km.get(r["name"])
        if stated is None:
            continue
        got = G.haversine(G.FORT[0], G.FORT[1], r["latitude"], r["longitude"])
        devs.append((abs(got - stated), r["name"], stated, got))
    devs.sort(reverse=True)
    print("\nlargest km-from-Fort disagreements among ACCEPTED railway rows:")
    for d, nm, stated, got in devs[:10]:
        print(f"  {d:5.2f}km  {nm:<34} source {stated:6.2f}  geocoded {got:6.2f}")

    print("\n" + "=" * 74)
    for cat, s in summary.items():
        n = len(s["rows"])
        print(f"{cat:<20} units={s['units']:>4} rows={n:>4} in_colombo={s['in']:>4} "
              f"failed={len(s['failures']):>3}  split={s['split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

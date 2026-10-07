"""Step 2j - One job: railway_stations, pvt_hospitals retries, pvt_med_centers,
fuel_stations (Ceypetco).

railway_stations
  Source : SLR "Station Details" (step2g).  263 in source, 104 within 60 km of
           Fort in scope.
  ADDED  : Colombo Fort.  The table measures distance FROM Fort, so Fort never
           appears as a row - its absence is an artefact of the table's
           construction, not a fact.  It is the busiest station in the country
           and sits in the Fort GN division.  Checked the other lines for the
           same artefact: Puttalam starts at Peralanda (17.1 km) and KV at
           Baseline Road (3.8 km), but both branch from stations that ARE
           listed (Ragama, Maradana), so no other origin is missing.
  Guards : distance-from-Fort agreement is PRIMARY (3 km).  Spelling variants
           are permitted for this category only, backstopped by that guard.

pvt_hospitals    - retries the 28 unresolved; address-first routing.
pvt_med_centers  - PHSRC MC rows, address-first routing.
fuel_stations    - Ceypetco only.  Lanka IOC incomplete (24 of ~202; the list
                   paginates 24/page with no controls, no select, no visible
                   API).  Laugfs not attempted.  Recorded as a coverage
                   limitation, not substituted.

Run:  python src/step2j_run_batch.py
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
RESULTS = ROOT / "results"

FORT_STATION = ("Colombo Fort Railway Station", "Fort, Colombo")


def main() -> int:
    rail = json.loads((REF / "railway_stations_source.json").read_text(encoding="utf-8"))
    rail_units, rail_km = [], {}
    for line, stations in rail["stations_within_max_km"].items():
        for s in stations:
            nm = f"{s['name']} Railway Station"
            rail_units.append((nm, s["name"]))
            rail_km[nm] = s["km_from_fort"]
    rail_units.insert(0, FORT_STATION)
    rail_km[FORT_STATION[0]] = 0.0
    G.RAILWAY_KM.update(rail_km)

    phs = json.loads((REF / "phsrc_source_lists.json").read_text(encoding="utf-8"))
    ph_units = [(r["name"], r["address"])
                for r in phs["categories"]["pvt_hospitals"]["rows"]]
    mc_units = [(r["name"], r["address"])
                for r in phs["categories"]["pvt_med_centers"]["rows"]]

    fuel = json.loads((REF / "fuel_stations_source.json").read_text(encoding="utf-8"))
    fuel_units = [(s["name"], s["raw_location"]) for s in fuel["stations"]]

    for cat, km in (("railway_stations", 2.0), ("pvt_hospitals", 2.0),
                    ("pvt_med_centers", 2.0), ("fuel_stations", 2.0)):
        G.ACCEPT_KM[cat] = km
    G.OVERPASS_TAGS["railway_stations"] = ['["railway"="station"]', '["railway"="halt"]']
    G.OVERPASS_TAGS["pvt_hospitals"] = ['["amenity"="hospital"]', '["amenity"="clinic"]']
    G.OVERPASS_TAGS["pvt_med_centers"] = ['["amenity"="clinic"]', '["amenity"="doctors"]']
    G.OVERPASS_TAGS["fuel_stations"] = ['["amenity"="fuel"]']
    G.ADDRESS_FIRST_CATEGORIES |= {"pvt_med_centers", "fuel_stations"}

    G.SOURCES["railway_stations"] = (
        "Sri Lanka Railways 'Station Details', railway.gov.lk "
        "(index.php?option=com_content&view=article&id=165&Itemid=191)",
        f"{rail['counts']['total']} stations in source; the "
        f"{rail['counts']['within_max_km']} within 60 km of Fort are in scope, "
        "plus Colombo Fort added manually (the table measures distance FROM "
        "Fort so Fort has no row; other lines branch from listed stations). "
        "PRIMARY acceptance test is agreement with the source's km-from-Fort "
        "(3 km tolerance). Spelling variants permitted for this category only, "
        "backstopped by that guard.")
    G.SOURCES["pvt_hospitals"] = (
        phs["source"] + " (partial listing also at phsrc.lk/pages_e.php?id=12)",
        "Registrations with prefix PHSRC/PH - the register's own named "
        "classification, not a residual bucket. De-duplicated on ADDRESS: "
        f"{phs['categories']['pvt_hospitals']['registrations']} registrations -> "
        f"{phs['categories']['pvt_hospitals']['rows_after_address_dedup']} rows.")
    G.SOURCES["pvt_med_centers"] = (
        phs["source"],
        "Registrations with prefix PHSRC/MC (Medical Centres) - the register's "
        "own named classification, not a residual bucket. De-duplicated on "
        f"ADDRESS: {phs['categories']['pvt_med_centers']['registrations']} "
        f"registrations -> "
        f"{phs['categories']['pvt_med_centers']['rows_after_address_dedup']} rows.")
    G.SOURCES["fuel_stations"] = (
        "; ".join(fuel["sources"].get("Ceypetco", [])),
        "Ceypetco district filling-station pages for Colombo and the districts "
        "that touch it (fuel stations carry the 10 km buffer). COVERAGE "
        "LIMITATION: Lanka IOC is incomplete (24 of ~202 captured; the dealer "
        "list paginates 24 per page with no controls, no select and no visible "
        "API) and Laugfs was not attempted. Neither was substituted.")

    polys = G.build_polygon_cache()
    cache = G.load_cache()
    session = requests.Session()
    session.headers.update({"User-Agent": G.UA})
    print(f"polygons {len(polys)} | cache {len(cache)} | "
          f"google={'yes' if os.environ.get('GOOGLE_MAPS_API_KEY') else 'NO'}")

    jobs = [("railway_stations", rail_units), ("pvt_hospitals", ph_units),
            ("pvt_med_centers", mc_units), ("fuel_stations", fuel_units)]
    all_units = [u for _, us in jobs for u in us]
    print(f"\nanchor pre-pass over {len(all_units)} units ...")
    for i, (nm, loc) in enumerate(all_units, 1):
        G.locality_centre(session, loc, cache)
        if i % 75 == 0:
            print(f"  anchors {i}/{len(all_units)}")
    collapsed = G.detect_anchor_collapse(cache)
    print(f"anchor collapses: {len(collapsed)}")

    summary = {}
    for cat, units in jobs:
        summary[cat] = G.write_category(cat, units, polys, cache, session,
                                        collapsed=collapsed)

    # ---- spelling corrections applied (railway only) ---------------------
    if G.SPELLING_LOG:
        lines = ["source_spelling | query_that_matched | km_source | km_geocoded | disagreement"]
        for s in G.SPELLING_LOG:
            lines.append(f"{s['source_spelling']} | {s['query_that_matched']} | "
                         f"{s['km_source']} | {s['km_geocoded']} | {s['disagreement_km']}")
        (RESULTS / "step2j_spelling_corrections.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        print(f"\nspelling corrections applied: {len(G.SPELLING_LOG)}")
        for s in G.SPELLING_LOG[:12]:
            print(f"  {s['source_spelling']:<32} -> {s['query_that_matched'][:44]:<46}"
                  f" dev={s['disagreement_km']}km")

    print("\n" + "=" * 76)
    for cat, s in summary.items():
        n = len(s["rows"])
        print(f"{cat:<20} units={s['units']:>4} rows={n:>4} in_colombo={s['in']:>4} "
              f"failed={len(s['failures']):>3}")
        print(f"{'':20} split={s['split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

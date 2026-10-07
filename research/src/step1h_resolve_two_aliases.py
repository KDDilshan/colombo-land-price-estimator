"""Step 1h - Resolve the two outstanding location references by bounded geocode.

Both are settled by geocoding bounded to the Colombo district bounding box and
testing the returned point against the 557 Colombo ADM4 polygons.  Neither is
resolved from the ikman Town tag: doing that would assign the division from the
tag, which is exactly the non-determinism R14 measures.

  1. Thalagala        query "Thalagala Road, Godagama, Sri Lanka"
                      -> alias to the containing GN division
  2. Kindelpitiya     query the locality bounded to Colombo
                      -> inside  : alias to the containing GN division
                      -> outside : the locality is in Kalutara (Bandaragama),
                                   so every spelling is added to the
                                   out-of-district list instead

Writes to data/reference/gn_alias_manual.csv and/or
data/reference/out_of_district_manual.csv, then step1b must be re-run.

Run:  python src/step1h_resolve_two_aliases.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step1c_geocode_unresolved import build_polygon_cache, locate  # noqa: E402

REF = ROOT / "data" / "reference"
URL = "https://nominatim.openstreetmap.org/search"
UA = "colombo-land-price-replication/1.0 (MSc replication)"

# Spelling variants seen in the adverts, resolved together.
THALAGALA_ALIASES = ["Thalagala", "Thalagala Road"]
KINDELPITIYA_ALIASES = [
    "Kindelpitiya", "kindelpitiya", "Kidelpitiya", "Kasbawa Kidelpitiya",
    "Kidelpitiya, Welmilla", "Kindelpitiya Kesbewa Bandaragama Rd",
]

TARGETS = [
    ("Thalagala", ["Thalagala, Sri Lanka",
                   "Thalagala Road, Godagama, Sri Lanka",
                   "Thalagala, Homagama, Sri Lanka"]),
    ("Kindelpitiya", ["Kindelpitiya, Sri Lanka",
                      "Kindelpitiya, Kesbewa, Sri Lanka",
                      "Kidelpitiya, Sri Lanka"]),
]

# A locality question must be answered by a locality, not by a road that
# happens to carry the locality's name.  "Kesbewa - Kindelpitiya - Bandaragama
# Road" runs from Colombo into Kalutara and its midpoint says nothing about
# where Kindelpitiya is.
PLACE_CLASSES = {"place", "boundary", "landuse"}


def is_place_hit(hit: dict) -> bool:
    return hit.get("class") in PLACE_CLASSES


def km_to_nearest_polygon(lon: float, lat: float, polys: list[dict]) -> float:
    """Great-circle distance from a point to the nearest Colombo ADM4 vertex."""
    import math

    best = float("inf")
    for poly in polys:
        x0, y0, x1, y1 = poly["bbox"]
        if min(abs(lon - x0), abs(lon - x1)) > 0.6 and not (x0 <= lon <= x1):
            continue
        for ring in poly["rings"]:
            for px, py in ring:
                dlat = math.radians(py - lat)
                dlon = math.radians(px - lon)
                a = (math.sin(dlat / 2) ** 2
                     + math.cos(math.radians(lat)) * math.cos(math.radians(py))
                     * math.sin(dlon / 2) ** 2)
                d = 2 * 6371.0 * math.asin(math.sqrt(a))
                if d < best:
                    best = d
    return best


def main() -> int:
    polys = build_polygon_cache()
    xs = [p["bbox"][0] for p in polys] + [p["bbox"][2] for p in polys]
    ys = [p["bbox"][1] for p in polys] + [p["bbox"][3] for p in polys]
    viewbox = f"{min(xs)},{max(ys)},{max(xs)},{min(ys)}"
    print(f"Colombo district viewbox: {viewbox}\n")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    def query(q: str, bounded: bool):
        params = {"q": q, "format": "json", "limit": 10,
                  "countrycodes": "lk", "addressdetails": 1}
        if bounded:
            params.update({"viewbox": viewbox, "bounded": 1})
        try:
            r = session.get(URL, params=params, timeout=45)
            return r.json() if r.status_code == 200 else []
        except Exception as exc:
            print(f"   ! {q!r}: {exc}")
            return []

    results = {}
    for name, queries in TARGETS:
        print("=" * 72)
        print(f"{name}")
        print("=" * 72)
        found, outside_evidence = None, None
        for q in queries:
            hits = query(q, bounded=True)
            time.sleep(1.1)
            place_hits = [h for h in hits if is_place_hit(h)]
            print(f"  bounded query {q!r}: {len(hits)} hit(s), "
                  f"{len(place_hits)} of place type")
            for h in hits:
                poly = locate(float(h["lon"]), float(h["lat"]), polys)
                tag = poly["base_name"] + f" ({poly['ds']})" if poly else "OUTSIDE Colombo ADM4"
                kind = "PLACE" if is_place_hit(h) else f"not-a-place ({h.get('class')})"
                print(f"     {float(h['lat']):.5f},{float(h['lon']):.5f}  {tag}  [{kind}]")
                print(f"        {h.get('display_name','')[:100]}")
            for h in place_hits:
                poly = locate(float(h["lon"]), float(h["lat"]), polys)
                if poly and found is None:
                    found = (poly, h)
                elif poly is None and outside_evidence is None:
                    d = km_to_nearest_polygon(float(h["lon"]), float(h["lat"]), polys)
                    outside_evidence = (h, d)
                    print(f"        -> {d:.2f} km from the nearest Colombo ADM4 boundary")
            if found:
                break

        if found is None:
            hits = query(f"{name}, Sri Lanka", bounded=False)
            time.sleep(1.1)
            print(f"  unbounded fallback: {len(hits)} hit(s)")
            for h in hits[:3]:
                addr = h.get("address") or {}
                print(f"     {addr.get('state_district') or addr.get('county')} | "
                      f"{h.get('display_name','')[:90]}")
            results[name] = {
                "inside": False,
                "district": (hits[0].get("address") or {}).get("state_district")
                if hits else None,
                "km_outside": outside_evidence[1] if outside_evidence else None,
                "display": outside_evidence[0].get("display_name", "")
                if outside_evidence else "",
            }
        else:
            poly, h = found
            results[name] = {"inside": True, "gn": poly["base_name"], "ds": poly["ds"],
                             "lat": float(h["lat"]), "lon": float(h["lon"]),
                             "display": h.get("display_name", "")}
        print()

    # ---------------------------------------------------------------- write
    alias_rows, ood_rows = [], []

    t = results["Thalagala"]
    if t["inside"]:
        for a in THALAGALA_ALIASES:
            alias_rows.append({
                "alias": a, "gn_division": t["gn"],
                "rationale": f"Bounded Nominatim query inside the Colombo district "
                             f"bounding box places Thalagala at {t['lat']:.5f},{t['lon']:.5f}, "
                             f"inside GN division {t['gn']} ({t['ds']}). "
                             f"OSM: {t['display'][:90]}",
            })
        print(f"Thalagala    -> alias to GN division {t['gn']} ({t['ds']})")
    else:
        print(f"Thalagala    -> NO bounded hit inside Colombo ADM4; "
              f"unbounded says {t.get('district')}")

    k = results["Kindelpitiya"]
    if k["inside"]:
        for a in KINDELPITIYA_ALIASES:
            alias_rows.append({
                "alias": a, "gn_division": k["gn"],
                "rationale": f"Bounded Nominatim query places Kindelpitiya at "
                             f"{k['lat']:.5f},{k['lon']:.5f}, inside GN division "
                             f"{k['gn']} ({k['ds']}). OSM: {k['display'][:90]}",
            })
        print(f"Kindelpitiya -> alias to GN division {k['gn']} ({k['ds']})")
    else:
        for a in KINDELPITIYA_ALIASES:
            if len(a.split()) > 2:
                continue  # the long forms carry a Colombo token and are handled by it
            ood_rows.append({
                "place": a,
                "district": k.get("district") or "Kalutara",
                "rationale": "Bounded Nominatim query against the Colombo district "
                             "bounding box returned no point inside any Colombo ADM4 "
                             "polygon; the locality sits on the Kesbewa/Bandaragama "
                             "boundary and Bandaragama is in Kalutara district.",
            })
        print(f"Kindelpitiya -> OUTSIDE Colombo; added to out_of_district_manual.csv")

    if alias_rows:
        path = REF / "gn_alias_manual.csv"
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame(
            columns=["alias", "gn_division", "rationale"])
        new = pd.concat([existing, pd.DataFrame(alias_rows)], ignore_index=True)
        new = new.drop_duplicates("alias", keep="last")
        new.to_csv(path, index=False, encoding="utf-8")
        print(f"\nWrote {path} ({len(new)} aliases)")

    if ood_rows:
        path = REF / "out_of_district_manual.csv"
        existing = pd.read_csv(path)
        new = pd.concat([existing, pd.DataFrame(ood_rows)], ignore_index=True)
        new = new.drop_duplicates("place", keep="last")
        new.to_csv(path, index=False, encoding="utf-8")
        print(f"Wrote {path} ({len(new)} entries)")

    print("\nNow re-run: step1b -> step1e -> step1g")
    return 0


if __name__ == "__main__":
    sys.exit(main())

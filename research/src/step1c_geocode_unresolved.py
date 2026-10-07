"""Step 1c - Resolve leftover location references by geocoding.

Section 3.2.2 of the dissertation resolves advertisements whose location is
given as a road name, a landmark or an informal town name by "analyzing the
road names and other location references mentioned in the advertisement" and
looking the place up on Google Maps.  This script is the reproducible
equivalent: every location string that steps 1b could not match is sent to
OpenStreetMap Nominatim, and the returned point is tested against the 557
Colombo ADM4 polygons.

  point inside a Colombo GN polygon -> alias to that GN division
  point elsewhere in Sri Lanka      -> the listing is outside Colombo district
  no result                         -> stays unmapped, reported by name

Nothing is guessed: an alias is only written when a geocoded point actually
falls inside a GN division polygon, and the OSM display name is stored next to
it so every alias can be audited.

Outputs
-------
data/reference/colombo_adm4_polygons.json  cached geometry (built once)
data/reference/gn_alias_geocoded.csv       alias -> GN division (+ evidence)
data/reference/geocode_out_of_district.csv places that geocode outside Colombo
data/reference/geocode_failed.csv          strings Nominatim could not resolve

Run:  python src/step1c_geocode_unresolved.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
RESULTS = ROOT / "results"

ADM4_SRC = ROOT / "data" / "raw" / "lka_admin_boundaries.geojson" / "lka_admin4.geojson"
POLY_CACHE = REF / "colombo_adm4_polygons.json"

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "colombo-land-price-replication/1.0 (MSc dissertation replication)"
SLEEP_SECONDS = 1.1  # Nominatim usage policy: at most 1 request per second

SINHALA_RE = re.compile(r"[඀-෿]")

# Strings that carry no place information; never sent to the geocoder.
JUNK_RE = re.compile(r"^(?:ls[\s\-]*\d+|[\d\s\-/,.]*)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Colombo ADM4 polygons
# ---------------------------------------------------------------------------
def build_polygon_cache() -> list[dict]:
    if POLY_CACHE.exists():
        return json.loads(POLY_CACHE.read_text(encoding="utf-8"))

    if not ADM4_SRC.exists():
        raise SystemExit(f"Missing {ADM4_SRC}. Run step0_fetch_gn_boundaries.py first.")

    print("building Colombo ADM4 polygon cache (this reads a large file once) ...")
    with open(ADM4_SRC, encoding="utf-8") as fh:
        gj = json.load(fh)

    raw = pd.read_csv(REF / "gn_divisions_raw_557.csv")
    base_of = dict(zip(raw.gn_pcode, raw.base_name))

    out = []
    for feat in gj["features"]:
        p = feat["properties"]
        if p.get("adm2_name") != "Colombo":
            continue
        geom = feat["geometry"]
        polys = (
            [geom["coordinates"]]
            if geom["type"] == "Polygon"
            else geom["coordinates"]
        )
        rings = [poly[0] for poly in polys]  # outer rings only
        xs = [pt[0] for r in rings for pt in r]
        ys = [pt[1] for r in rings for pt in r]
        out.append(
            {
                "pcode": p["adm4_pcode"],
                "name": p["adm4_name"],
                "base_name": base_of.get(p["adm4_pcode"]),
                "ds": p["adm3_name"],
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "rings": rings,
            }
        )

    POLY_CACHE.write_text(json.dumps(out), encoding="utf-8")
    print(f"  cached {len(out)} polygons -> {POLY_CACHE.name}")
    return out


def point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Standard ray-casting test."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def locate(lon: float, lat: float, polys: list[dict]):
    for poly in polys:
        x0, y0, x1, y1 = poly["bbox"]
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        for ring in poly["rings"]:
            if point_in_ring(lon, lat, ring):
                return poly
    return None


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def geocode(query: str, session: requests.Session):
    params = {
        "q": query if "sri lanka" in query.lower() else f"{query}, Sri Lanka",
        "format": "json",
        "limit": 1,
        "countrycodes": "lk",
        "addressdetails": 1,
    }
    try:
        r = session.get(NOMINATIM, params=params, timeout=45)
        if r.status_code != 200:
            return None
        js = r.json()
        if not js:
            return None
        top = js[0]
        return {
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
            "display_name": top.get("display_name", ""),
            "osm_district": (top.get("address") or {}).get("state_district")
            or (top.get("address") or {}).get("county")
            or "",
        }
    except Exception as exc:  # network hiccup - reported, not fatal
        print(f"    ! {query!r}: {exc}")
        return None


def candidate_strings() -> pd.DataFrame:
    """Location strings still unmapped after the last step-1b run."""
    path = RESULTS / "step1_unmapped_listings.csv"
    if not path.exists():
        raise SystemExit(
            "results/step1_unmapped_listings.csv not found.\n"
            "Run: python src/step1b_map_listings_to_gn.py first."
        )
    um = pd.read_csv(path)

    rows = []
    for col in ("Address_Raw", "Town"):
        if col not in um.columns:
            continue
        vc = um[col].dropna().astype(str).str.strip()
        vc = vc[vc != ""]
        for text, n in vc.value_counts().items():
            if JUNK_RE.match(text):
                continue
            rows.append({"query": text, "n_listings": int(n), "from_field": col})

    df = pd.DataFrame(rows)
    # De-duplicate case-insensitively, keeping the most frequent spelling.
    df["key"] = df["query"].str.lower().str.strip()
    df = (
        df.sort_values("n_listings", ascending=False)
        .drop_duplicates("key")
        .reset_index(drop=True)
    )
    return df


def main() -> int:
    REF.mkdir(parents=True, exist_ok=True)
    polys = build_polygon_cache()
    print(f"Colombo ADM4 polygons        : {len(polys)}")

    todo = candidate_strings()
    print(f"distinct strings to geocode  : {len(todo)}")
    print(f"estimated time               : {len(todo) * SLEEP_SECONDS / 60:.1f} min\n")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    inside, outside, failed = [], [], []
    for i, row in enumerate(todo.itertuples(), start=1):
        res = geocode(row.query, session)
        time.sleep(SLEEP_SECONDS)
        if res is None:
            failed.append({"alias": row.query, "n_listings": row.n_listings})
            print(f"[{i:3d}/{len(todo)}] {row.query[:48]:48s} -> no result")
            continue

        poly = locate(res["lon"], res["lat"], polys)
        if poly is None:
            outside.append(
                {
                    "alias": row.query,
                    "n_listings": row.n_listings,
                    "lat": res["lat"],
                    "lon": res["lon"],
                    "osm_district": res["osm_district"],
                    "display_name": res["display_name"],
                }
            )
            print(
                f"[{i:3d}/{len(todo)}] {row.query[:48]:48s} -> OUTSIDE Colombo "
                f"({res['osm_district']})"
            )
            continue

        inside.append(
            {
                "alias": row.query,
                "gn_division": poly["base_name"],
                "n_listings": row.n_listings,
                "adm4_name": poly["name"],
                "ds_division": poly["ds"],
                "geocoded_lat": res["lat"],
                "geocoded_lon": res["lon"],
                "osm_display_name": res["display_name"],
                "rationale": "Nominatim point falls inside this GN division polygon",
            }
        )
        print(f"[{i:3d}/{len(todo)}] {row.query[:48]:48s} -> {poly['base_name']}")

    pd.DataFrame(inside).to_csv(REF / "gn_alias_geocoded.csv", index=False, encoding="utf-8")
    pd.DataFrame(outside).to_csv(
        REF / "geocode_out_of_district.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(failed).to_csv(REF / "geocode_failed.csv", index=False, encoding="utf-8")

    n_in = sum(r["n_listings"] for r in inside)
    n_out = sum(r["n_listings"] for r in outside)
    n_fail = sum(r["n_listings"] for r in failed)
    print("\n" + "=" * 60)
    print(f"resolved inside Colombo : {len(inside):4d} strings / {n_in:5d} listings")
    print(f"outside Colombo         : {len(outside):4d} strings / {n_out:5d} listings")
    print(f"not geocodable          : {len(failed):4d} strings / {n_fail:5d} listings")
    print("\nRe-run step1b_map_listings_to_gn.py to apply these aliases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

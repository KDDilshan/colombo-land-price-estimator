"""Step 6d (Ratnapura variant) - distance to running water, from OpenStreetMap
via Overpass.

Identical method to src/step6d_osm_waterways.py - same waterway filter
(river|stream|canal), same nearest-VERTEX haversine distance, same R_EARTH, same
4-dp rounding. Three changes, all of them necessary and all recorded:

  1. BBOX. The Colombo box (6.65, 79.75, 7.10, 80.30) does NOT contain these 10
     points - Kalawana sits at 6.5311 N (south of it) and Belihuloya at
     80.7720 E (east of it). Querying it would silently return no waterway near
     either point and report a falsely large distance. The new box is the extent
     of the 10 points plus a uniform 0.15 deg margin, which is slightly more
     generous than the 0.08-0.13 deg margin the Colombo box carried over its own
     centroids. Every point is asserted inside the box before the query runs.

  2. CACHE FILE. Writes data/reference/osm_waterways_ratnapura.json. The Colombo
     cache is neither read nor overwritten.

  3. VECTORISED DISTANCE. The Colombo script loops in pure Python over every
     vertex; this box returns far more geometry, so the same haversine is
     evaluated with numpy instead. Same formula, same result, no method change.

dist_to_kelani_km is EXPECTED TO BE EMPTY here. The Kelani Ganga does not drain
Sabaragamuwa - Ratnapura sits on the Kalu Ganga - so no way named Kelani exists
inside this box. A blank column is the correct and honest output, not a failure.

Input : data/processed/ratnapura_10pt_points.csv
Output: data/processed/ratnapura_10pt_water.csv
"""

import csv
import json
import math
import os
import time
from collections import Counter

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS = os.path.join(ROOT, "data", "processed", "ratnapura_10pt_points.csv")
CACHE = os.path.join(ROOT, "data", "reference", "osm_waterways_ratnapura.json")
OUT = os.path.join(ROOT, "data", "processed", "ratnapura_10pt_water.csv")

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]

# south, west, north, east - extent of the 10 points + 0.15 deg on every side
BBOX = (6.38, 80.11, 7.01, 80.93)
MARGIN_DEG = 0.15
R_EARTH = 6371.0088

QUERY = f"""[out:json][timeout:240];
(
  way["waterway"~"^(river|stream|canal)$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;"""


def check_bbox(pts):
    """Refuse to query a box that does not contain every point with margin."""
    s, w, n, e = BBOX
    print(f"BBOX (south, west, north, east) = {BBOX}")
    print(f"  spans {n - s:.3f} deg lat x {e - w:.3f} deg lon "
          f"(~{(n - s) * 110.6:.0f} km x {(e - w) * 110.6 * math.cos(math.radians(6.7)):.0f} km)\n")
    lats = [float(p["Latitude"]) for p in pts]
    lons = [float(p["Longitude"]) for p in pts]
    print(f"  point extent : lat {min(lats):.4f}..{max(lats):.4f}  "
          f"lon {min(lons):.4f}..{max(lons):.4f}")
    print(f"  margins      : S {min(lats) - s:.4f}  N {n - max(lats):.4f}  "
          f"W {min(lons) - w:.4f}  E {e - max(lons):.4f} deg")
    bad = [p["Label"] for p in pts
           if not (s < float(p["Latitude"]) < n and w < float(p["Longitude"]) < e)]
    if bad:
        raise SystemExit(f"BBOX does not contain: {bad}")
    tight = min(min(lats) - s, n - max(lats), min(lons) - w, e - max(lons))
    if tight < MARGIN_DEG - 1e-9:
        raise SystemExit(f"margin {tight:.4f} deg is below the required {MARGIN_DEG}")
    print(f"  all {len(pts)} points inside, tightest margin {tight:.4f} deg "
          f"(>= {MARGIN_DEG}) - OK\n")


def haversine_np(lat1, lon1, lats, lons):
    p1 = math.radians(lat1)
    p2 = np.radians(lats)
    a = (np.sin((p2 - p1) / 2.0) ** 2
         + math.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon1) / 2.0) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def fetch():
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 1000:
        print(f"using cached Overpass response ({os.path.getsize(CACHE) / 1e6:.1f} MB)")
        return json.load(open(CACHE, encoding="utf-8"))
    last = None
    for url in ENDPOINTS:
        try:
            print(f"querying {url}")
            t0 = time.perf_counter()
            r = requests.post(url, data={"data": QUERY}, timeout=300,
                              headers={"User-Agent": "colombo-land-price-replication/1.0"})
            r.raise_for_status()
            doc = r.json()
            print(f"  {time.perf_counter() - t0:.1f}s, {len(r.content) / 1e6:.1f} MB")
            with open(CACHE, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            return doc
        except Exception as exc:
            print(f"  ! {type(exc).__name__}: {str(exc)[:120]}")
            last = exc
            time.sleep(5)
    raise SystemExit(f"Overpass unreachable: {last}")


def main():
    with open(POINTS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))
    check_bbox(pts)

    doc = fetch()
    ways = [e for e in doc.get("elements", []) if e.get("type") == "way" and e.get("geometry")]
    print(f"{len(ways)} waterway ways")

    all_lat, all_lon, kel_lat, kel_lon = [], [], [], []
    kelani_ways = 0
    named = Counter()
    for wobj in ways:
        tags = wobj.get("tags", {})
        name = " ".join(str(v) for k, v in tags.items() if k.startswith("name"))
        la = [g["lat"] for g in wobj["geometry"]]
        lo = [g["lon"] for g in wobj["geometry"]]
        all_lat.extend(la)
        all_lon.extend(lo)
        if tags.get("waterway") == "river" and tags.get("name"):
            named[tags["name"]] += len(la)
        if "kelani" in name.lower() and tags.get("waterway") == "river":
            kel_lat.extend(la)
            kel_lon.extend(lo)
            kelani_ways += 1

    print(f"  vertices: {len(all_lat)} total, {len(kel_lat)} on "
          f"{kelani_ways} Kelani Ganga ways")
    if not kel_lat:
        print("  ! no way named Kelani found - dist_to_kelani_km will be blank")
    print("\n  named rivers in this box, by vertex count (diagnostic only):")
    for nm, c in named.most_common(10):
        print(f"    {nm:40s} {c:6d}")
    print()

    all_lat = np.asarray(all_lat)
    all_lon = np.asarray(all_lon)
    kel_lat = np.asarray(kel_lat)
    kel_lon = np.asarray(kel_lon)

    rows = []
    for p in pts:
        lat, lon = float(p["Latitude"]), float(p["Longitude"])
        d_all = float(haversine_np(lat, lon, all_lat, all_lon).min()) if all_lat.size else None
        d_kel = float(haversine_np(lat, lon, kel_lat, kel_lon).min()) if kel_lat.size else None
        rows.append({"Label": p["Label"],
                     "Latitude": f"{lat:.6f}", "Longitude": f"{lon:.6f}",
                     "dist_to_stream_km": "" if d_all is None else round(d_all, 4),
                     "dist_to_kelani_km": "" if d_kel is None else round(d_kel, 4)})

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT}\n")
    for c in ("dist_to_stream_km", "dist_to_kelani_km"):
        v = [float(r[c]) for r in rows if r[c] != ""]
        if v:
            print(f"{c:20s} min={min(v):7.3f} max={max(v):7.3f} "
                  f"mean={sum(v) / len(v):7.3f} distinct={len(set(v))} "
                  f"nulls={len(rows) - len(v)}")
        else:
            print(f"{c:20s} all {len(rows)} blank (expected - see module docstring)")


if __name__ == "__main__":
    main()

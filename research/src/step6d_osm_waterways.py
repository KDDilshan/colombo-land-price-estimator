"""Step 6d - distance to running water, from OpenStreetMap via Overpass.

  dist_to_stream_km   nearest waterway of any kind (river, stream, canal)
  dist_to_kelani_km   nearest segment of the Kelani Ganga specifically - the
                      river that floods Kolonnawa, Kaduwela and Kotikawatta

METHOD AND ITS LIMIT.  Overpass returns each waterway as an ordered list of
vertices. Distance is measured to the nearest VERTEX, not to the nearest point
on the segment between two vertices. OSM river geometry in this area is dense
(tens of metres between vertices), so the error is well under the ~100 m that
would matter at a 1 km feature scale, but it is an approximation and it is
recorded as one.

Output: data/processed/osm_water_194.csv
"""

import csv
import json
import math
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
CACHE = os.path.join(ROOT, "data", "reference", "osm_waterways_colombo.json")
OUT = os.path.join(ROOT, "data", "processed", "osm_water_194.csv")

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
BBOX = (6.65, 79.75, 7.10, 80.30)          # south, west, north, east
R_EARTH = 6371.0088

QUERY = f"""[out:json][timeout:240];
(
  way["waterway"~"^(river|stream|canal)$"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;"""


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def fetch():
    if os.path.exists(CACHE) and os.path.getsize(CACHE) > 1000:
        print(f"using cached Overpass response ({os.path.getsize(CACHE) / 1e6:.1f} MB)")
        return json.load(open(CACHE, encoding="utf-8"))
    last = None
    for url in ENDPOINTS:
        try:
            print(f"querying {url}")
            r = requests.post(url, data={"data": QUERY}, timeout=300,
                              headers={"User-Agent": "colombo-land-price-replication/1.0"})
            r.raise_for_status()
            doc = r.json()
            with open(CACHE, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            return doc
        except Exception as exc:
            print(f"  ! {type(exc).__name__}: {str(exc)[:120]}")
            last = exc
            time.sleep(5)
    raise SystemExit(f"Overpass unreachable: {last}")


def main():
    doc = fetch()
    ways = [e for e in doc.get("elements", []) if e.get("type") == "way" and e.get("geometry")]
    print(f"{len(ways)} waterway ways")

    all_pts, kelani_pts = [], []
    kelani_ways = 0
    for w in ways:
        tags = w.get("tags", {})
        name = " ".join(str(v) for k, v in tags.items() if k.startswith("name"))
        pts = [(g["lat"], g["lon"]) for g in w["geometry"]]
        all_pts.extend(pts)
        if "kelani" in name.lower() and tags.get("waterway") == "river":
            kelani_pts.extend(pts)
            kelani_ways += 1
    print(f"  vertices: {len(all_pts)} total, {len(kelani_pts)} on "
          f"{kelani_ways} Kelani Ganga ways")
    if not kelani_pts:
        print("  ! no way named Kelani found - dist_to_kelani_km will be blank")

    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        centroids = list(csv.DictReader(fh))

    rows = []
    for p in centroids:
        lat, lon = float(p["gn_lat"]), float(p["gn_lon"])
        d_all = min(haversine(lat, lon, a, b) for a, b in all_pts) if all_pts else None
        d_kel = min(haversine(lat, lon, a, b) for a, b in kelani_pts) if kelani_pts else None
        rows.append({"Address": p["Address"], "Address_ID": p["Address_ID"],
                     "dist_to_stream_km": "" if d_all is None else round(d_all, 4),
                     "dist_to_kelani_km": "" if d_kel is None else round(d_kel, 4)})

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT}\n")
    for c in ("dist_to_stream_km", "dist_to_kelani_km"):
        v = [float(r[c]) for r in rows if r[c] != ""]
        if v:
            print(f"{c:20s} min={min(v):7.3f} max={max(v):7.3f} "
                  f"mean={sum(v) / len(v):7.3f} distinct={len(set(v))} "
                  f"nulls={len(rows) - len(v)}")


if __name__ == "__main__":
    main()

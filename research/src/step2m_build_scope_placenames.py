"""Step 2m - Build the place-name list used to filter units BEFORE geocoding.

Every unit in the new categories arrives with a postal address.  Geocoding a
Kandy branch to find out it is in Kandy is pure cost, so an address that names
no Colombo-district or buffer place is dropped before any geocoder is called.

The list is DERIVED, not hand-written.  From the national ADM4 (GN division)
boundaries:

  in-district : every GN division and DS division in Colombo district
  buffer      : every GN division in any other district whose polygon comes
                within BUFFER_KM of the Colombo district boundary

plus the postal forms "Colombo 1".."Colombo 15" (with and without a leading
zero), which addresses use far more often than a GN name.

Dropping a unit is a filter on the ADDRESS TEXT only.  It never decides where a
unit is - point-in-polygon still does that after geocoding.  A false negative
costs a row; a false positive costs one wasted geocode.  The buffer is
therefore generous.

Output: data/reference/colombo_scope_placenames.json

Run:  python src/step2m_build_scope_placenames.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
ADM4 = ROOT / "data" / "raw" / "lka_admin_boundaries.geojson" / "lka_admin4.geojson"

BUFFER_KM = 12.0     # amenity buffer is 10 km; 2 km of slack on a name filter


def rings_of(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def main() -> int:
    gj = json.loads(ADM4.read_text(encoding="utf-8"))
    print(f"{len(gj['features'])} GN divisions nationally")

    cmb_pts, cmb_names, ds_names, others = [], set(), set(), []
    for f in gj["features"]:
        p = f["properties"]
        rings = rings_of(f["geometry"])
        pts = [(pt[1], pt[0]) for r in rings for pt in r[::5]]
        if not pts:
            continue
        if p.get("adm2_name") == "Colombo":
            cmb_pts.extend(pts)
            cmb_names.add(p["adm4_name"])
            ds_names.add(p["adm3_name"])
        else:
            others.append((p["adm4_name"], p["adm3_name"], p["adm2_name"], pts))

    # Coarse pre-filter on a bounding box, then the real distance test.
    lats = [a for a, _ in cmb_pts]
    lons = [b for _, b in cmb_pts]
    pad = BUFFER_KM / 100.0
    box = (min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad)
    cmb_thin = cmb_pts[::3]

    buf_names, buf_ds = set(), set()
    for name, ds, dist, pts in others:
        near = [q for q in pts
                if box[0] <= q[0] <= box[1] and box[2] <= q[1] <= box[3]]
        if not near:
            continue
        hit = any(haversine(a, b, c, d) <= BUFFER_KM
                  for a, b in near[::2] for c, d in cmb_thin[::4])
        if hit:
            buf_names.add(name)
            buf_ds.add(f"{ds} ({dist})")

    postal = [f"Colombo {n}" for n in range(1, 16)]
    postal += [f"Colombo 0{n}" for n in range(1, 10)]

    def norm(s):
        return " ".join(str(s).replace("-", " ").split()).strip().lower()

    tokens = sorted({norm(x) for x in
                     cmb_names | set(ds_names) | buf_names | set(postal)
                     if x and len(str(x)) > 2})

    doc = {"buffer_km": BUFFER_KM,
           "colombo_gn_divisions": len(cmb_names),
           "colombo_ds_divisions": sorted(ds_names),
           "buffer_gn_divisions": len(buf_names),
           "buffer_ds_divisions": sorted(buf_ds),
           "postal_forms": postal,
           "tokens": tokens}
    (REF / "colombo_scope_placenames.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Colombo GN {len(cmb_names)} | buffer GN {len(buf_names)} "
          f"| DS in buffer {len(buf_ds)} | tokens {len(tokens)}")
    print("buffer DS divisions:", ", ".join(sorted(buf_ds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

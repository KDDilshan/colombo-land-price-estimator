"""Build deploy/gn_boundaries.geojson - the click -> GN division lookup.

The model data carries no coordinates, only division names. Nothing here is
invented: the geometry and the name mapping both come from files this project
already produced.

  data/reference/colombo_adm4_polygons.json   557 ADM4 (GN division) polygons for
                                              Colombo district, each carrying the
                                              `base_name` with its directional
                                              suffix (North/South/East/...) stripped
  data/reference/gn_lookup.csv                the merged division table the model's
                                              own coordinates come from: one row per
                                              merged division, with area_sqkm and
                                              the lat/lon Step 4/6a used
  data/processed/gn_centroids.csv             the 194 modelled divisions and their
                                              representative points

Divisions are assembled by matching Address to `base_name`, falling back to the
full `name` (only 'havelock town' needs that - its base_name is 'Havelock').

Note on the merge rule. `gn_divisions_merged_399.csv` also has a `merged_from`
column, but it is NOT used here: for the six divisions whose name occurs in more
than one DS division (dampe, gangodavila, kawdana, koswatta, kurunduwatta,
udumulla) it lists only one DS's members, while the area_sqkm and lat/lon in
gn_lookup.csv - the numbers the feature extraction actually ran on - span all of
them. Matching on base_name reproduces gn_lookup's area for all 194 divisions to
within 0.5%, so that is the rule the model was built with. The check is asserted
below rather than trusted.

Member polygons are emitted as one MultiPolygon. They are not geometrically
unioned - interior edges are shared, so "inside any member ring" is already the
correct point-in-polygon test for the dissolved division.

The remaining Colombo ADM4 divisions are written too, flagged modelled=false, so a
click outside the 194 can still be named before it falls through to the
Colombo-wide average path.

Run:  python build_boundaries.py
"""

import csv
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLYGONS = os.path.join(ROOT, "data", "reference", "colombo_adm4_polygons.json")
LOOKUP = os.path.join(ROOT, "data", "reference", "gn_lookup.csv")
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
ENCODING = os.path.join(HERE, "deploy", "address_encoding.csv")
OUT = os.path.join(HERE, "deploy", "gn_boundaries.geojson")

PRECISION = 6          # ~0.11 m at this latitude
EARTH_RADIUS_M = 6371008.8
AREA_TOLERANCE = 0.01  # spherical vs geodesic area differs by a systematic ~0.43%


def ring_bbox(ring):
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    return [min(xs), min(ys), max(xs), max(ys)]


def merge_bbox(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def spherical_area_km2(rings):
    total = 0.0
    for ring in rings:
        pts = ring + [ring[0]]
        s = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            s += (math.radians(x2) - math.radians(x1)) * (
                2 + math.sin(math.radians(y1)) + math.sin(math.radians(y2)))
        total += abs(s * EARTH_RADIUS_M * EARTH_RADIUS_M / 2)
    return total / 1e6


def rounded(rings):
    coords, boxes = [], []
    for ring in rings:
        coords.append([[[round(c[0], PRECISION), round(c[1], PRECISION)]
                        for c in ring]])
        boxes.append(ring_bbox(ring))
    return coords, boxes


def main():
    with open(POLYGONS, encoding="utf-8") as fh:
        polys = json.load(fh)
    with open(LOOKUP, encoding="utf-8-sig") as fh:
        lookup = {r["gn_division"].strip().lower(): r for r in csv.DictReader(fh)}
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        centroids = list(csv.DictReader(fh))
    with open(ENCODING, encoding="utf-8-sig") as fh:
        encoded = {r["Address"].strip().lower() for r in csv.DictReader(fh)}

    addresses = {r["Address"].strip().lower(): r for r in centroids}

    # ------------------------------------------- resolve each division's polygons
    by_name = defaultdict(list)
    for p in polys:
        keys = {p["base_name"].strip().lower(), p["name"].strip().lower()}
        for k in keys:
            by_name[k].append(p)

    members, claimed = {}, set()
    for address in addresses:
        mem = by_name.get(address, [])
        members[address] = mem
        claimed.update(p["pcode"] for p in mem)

    unmodelled = [p for p in polys if p["pcode"] not in claimed]

    # ------------------------------------------------------------------- checks
    enc_only = sorted(encoded - set(addresses))
    cen_only = sorted(set(addresses) - encoded)

    # cen_only: gn_centroids.csv (a shared pipeline artifact, not retrained here)
    # still has a coordinate for this division, but the current model retrain's
    # outlier trimming dropped every one of its listings -- it's a real GN
    # division, just no longer priceable this run. Demote it to modelled=false
    # (same treatment as any other out-of-model ADM4 polygon) instead of failing
    # the whole build. enc_only -- the model claims a division with no coordinate
    # at all -- is a genuine data bug and stays fatal below.
    demoted = {a: members.pop(a) for a in cen_only}
    modelled_addrs = set(members.keys())

    no_polygon = sorted(a for a, m in members.items() if not m)

    # no ADM4 polygon may be claimed by two different divisions
    owner = {}
    collisions = []
    for address, mem in members.items():
        for p in mem:
            if p["pcode"] in owner and owner[p["pcode"]] != address:
                collisions.append((p["name"], owner[p["pcode"]], address))
            owner[p["pcode"]] = address

    # the assembled geometry must reproduce the area the model's own table records
    area_bad, missing_area = [], []
    for address, mem in members.items():
        rec = lookup.get(address)
        if rec is None:
            missing_area.append(address)
            continue
        got = spherical_area_km2([r for p in mem for r in p["rings"]])
        want = float(rec["area_sqkm"])
        if want > 0 and abs(got - want) / want > AREA_TOLERANCE:
            area_bad.append(f"{address} built={got:.3f} lookup={want:.3f} km2")

    multi_ds = sorted(a for a in addresses
                      if int(lookup[a]["n_ds_divisions"]) > 1) if not missing_area else []

    print(f"ADM4 polygons read            : {len(polys)}")
    print(f"divisions in gn_centroids.csv : {len(addresses)}")
    print(f"divisions in address_encoding : {len(encoded)}")
    print(f"modelled == encoding          : {modelled_addrs == encoded}  "
          f"({len(modelled_addrs)}/{len(encoded)})")
    print(f"divisions with no polygon     : {len(no_polygon)} {no_polygon}")
    print(f"in encoding, not in centroids : {len(enc_only)} {enc_only}")
    print(f"in centroids, not in encoding : {len(cen_only)} {cen_only}  (demoted to modelled=false)")
    print(f"polygons claimed twice        : {len(collisions)} {collisions}")
    print(f"not in gn_lookup.csv          : {len(missing_area)} {missing_area}")
    print(f"area disagrees with gn_lookup : {len(area_bad)} {area_bad}")
    print(f"polygons assigned / unmodelled: {len(claimed)} / {len(unmodelled)}")
    print(f"divisions spanning >1 DS      : {len(multi_ds)} {multi_ds}")

    if no_polygon or enc_only or collisions or missing_area or area_bad or modelled_addrs != encoded:
        raise SystemExit("boundary build failed its checks - refusing to write output")

    # -------------------------------------------------------------- build output
    features = []
    for address, row in sorted(addresses.items(), key=lambda kv: int(kv[1]["Address_ID"])):
        if address in demoted:
            continue
        mem = members[address]
        coords, boxes = rounded([r for p in mem for r in p["rings"]])
        features.append({
            "type": "Feature",
            "properties": {
                "address": address,
                "address_id": int(row["Address_ID"]),
                "modelled": True,
                "gn_lat": float(row["gn_lat"]),
                "gn_lon": float(row["gn_lon"]),
                "adm4_members": sorted(p["name"] for p in mem),
                "adm4_pcodes": sorted(p["pcode"] for p in mem),
                "ds_divisions": sorted({p["ds"] for p in mem}),
                "bbox": [round(v, PRECISION) for v in merge_bbox(boxes)],
            },
            "geometry": {"type": "MultiPolygon", "coordinates": coords},
        })

    for address, mem in sorted(demoted.items()):
        row = addresses[address]
        coords, boxes = rounded([r for p in mem for r in p["rings"]])
        features.append({
            "type": "Feature",
            "properties": {
                "address": address,
                "address_id": int(row["Address_ID"]),
                "modelled": False,
                "modelled_reason": "dropped by outlier trimming in the current model retrain",
                "gn_lat": float(row["gn_lat"]),
                "gn_lon": float(row["gn_lon"]),
                "adm4_members": sorted(p["name"] for p in mem),
                "adm4_pcodes": sorted(p["pcode"] for p in mem),
                "ds_divisions": sorted({p["ds"] for p in mem}),
                "bbox": [round(v, PRECISION) for v in merge_bbox(boxes)],
            },
            "geometry": {"type": "MultiPolygon", "coordinates": coords},
        })

    for p in sorted(unmodelled, key=lambda x: x["pcode"]):
        coords, boxes = rounded(p["rings"])
        features.append({
            "type": "Feature",
            "properties": {
                "address": p["name"].strip().lower(),
                "address_id": None,
                "modelled": False,
                "adm4_members": [p["name"]],
                "adm4_pcodes": [p["pcode"]],
                "ds_divisions": [p["ds"]],
                "bbox": [round(v, PRECISION) for v in merge_bbox(boxes)],
            },
            "geometry": {"type": "MultiPolygon", "coordinates": coords},
        })

    out = {
        "type": "FeatureCollection",
        "name": "gn_boundaries",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "description": (f"Colombo district GN divisions. modelled=true are the "
                        f"{len(modelled_addrs)} divisions the current Random Forest "
                        f"was trained on; modelled=false are the remaining ADM4 "
                        f"divisions plus any division {sorted(demoted)} that the "
                        f"model retrain no longer covers, named for context only."),
        "features": features,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)

    print(f"\nwrote {OUT}")
    print(f"  features: {len(features)} "
          f"({sum(f['properties']['modelled'] for f in features)} modelled)")
    print(f"  size    : {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

"""
Step 3 - clean every existing amenity coordinate file.

Rules (applied in this order, per file):
  1. drop rows whose latitude/longitude do not parse as numbers
  2. drop rows outside the Sri Lanka bounding box
  3. drop duplicate names   (case/whitespace-insensitive, first occurrence kept)
  4. drop duplicate coordinates (rounded to 5 dp ~ 1 m, first occurrence kept)
Everything that survives is kept - no category is dropped for being incomplete.

`govtschools.csv` is additionally split with the `zone_class` field carried by
`data/reference/schools_source.json`:
    national + provincial -> govtschools.csv   (used for BOTH Class A and Class B)
    semi_government       -> semigovtschools.csv

Outputs: data/amenities/clean/*.csv  and  results/step3_amenity_cleaning_report.txt
"""

import csv
import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "amenities")
OUT_DIR = os.path.join(SRC_DIR, "clean")
REPORT = os.path.join(ROOT, "results", "step3_amenity_cleaning_report.txt")
SCHOOLS_SRC = os.path.join(ROOT, "data", "reference", "schools_source.json")

# Sri Lanka bounding box (generous: mainland + islands)
LAT_MIN, LAT_MAX = 5.7, 10.0
LON_MIN, LON_MAX = 79.3, 82.1

FIELDS = ["name", "latitude", "longitude", "in_colombo_district"]


def norm(s):
    return " ".join((s or "").split()).lower()


def clean_rows(rows, stats):
    """Apply the four rules. `rows` are dicts with the standard four fields."""
    seen_name, seen_coord = set(), set()
    out = []
    for r in rows:
        stats["read"] += 1
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
        except (TypeError, ValueError):
            stats["bad_coord"] += 1
            continue
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            stats["out_of_box"] += 1
            stats["out_of_box_rows"].append(f"{r['name']} ({lat},{lon})")
            continue
        nk = norm(r["name"])
        if nk in seen_name:
            stats["dup_name"] += 1
            stats["dup_name_rows"].append(r["name"])
            continue
        ck = (round(lat, 5), round(lon, 5))
        if ck in seen_coord:
            stats["dup_coord"] += 1
            stats["dup_coord_rows"].append(f"{r['name']} @ {ck[0]},{ck[1]}")
            continue
        seen_name.add(nk)
        seen_coord.add(ck)
        out.append({
            "name": r["name"],
            "latitude": f"{lat:.7f}",
            "longitude": f"{lon:.7f}",
            "in_colombo_district": r.get("in_colombo_district", "") or "0",
        })
    stats["kept"] = len(out)
    return out


def new_stats():
    return {"read": 0, "bad_coord": 0, "out_of_box": 0, "dup_name": 0,
            "dup_coord": 0, "kept": 0, "out_of_box_rows": [],
            "dup_name_rows": [], "dup_coord_rows": []}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    files = sorted(f for f in os.listdir(SRC_DIR)
                   if f.endswith(".csv") and not f.startswith("_"))

    # zone_class lookup for the schools split
    zone = {}
    with open(SCHOOLS_SRC, encoding="utf-8") as fh:
        for r in json.load(fh)["rows"]:
            zone[norm(r["name"])] = r["zone_class"]

    report = OrderedDict()

    for fn in files:
        with open(os.path.join(SRC_DIR, fn), encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))

        if fn == "govtschools.csv":
            groups = {
                "govtschools.csv": [r for r in rows
                                    if zone.get(norm(r["name"])) in ("national", "provincial")],
                "semigovtschools.csv": [r for r in rows
                                        if zone.get(norm(r["name"])) == "semi_government"],
            }
            unknown = [r for r in rows if norm(r["name"]) not in zone]
            if unknown:
                groups["govtschools.csv"].extend(unknown)
        else:
            groups = {fn: rows}

        for out_name, grp in groups.items():
            st = new_stats()
            cleaned = clean_rows(grp, st)
            with open(os.path.join(OUT_DIR, out_name), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(cleaned)
            st["in_colombo"] = sum(1 for r in cleaned
                                   if str(r["in_colombo_district"]).strip() in ("1", "True", "true"))
            st["source_file"] = fn
            report[out_name] = st

    # supermarkets - no source data exists at all; write an empty file so the
    # downstream build has something explicit to read rather than a missing path
    sm = os.path.join(OUT_DIR, "supermarkets.csv")
    with open(sm, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
    st = new_stats()
    st["source_file"] = "(none - see data/reference/supermarkets_source.json)"
    st["in_colombo"] = 0
    report["supermarkets.csv"] = st

    lines = ["STEP 3 - AMENITY FILE CLEANING", "=" * 70, "",
             "Rules: invalid coords dropped, Sri Lanka bbox "
             f"({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E), duplicate names dropped,",
             "duplicate coordinates (5 dp) dropped. First occurrence always kept.", "",
             f"{'output file':26s} {'read':>5s} {'bad':>4s} {'obox':>5s} {'dupN':>5s} "
             f"{'dupC':>5s} {'kept':>5s} {'inCMB':>6s}"]
    for name, st in report.items():
        lines.append(f"{name:26s} {st['read']:5d} {st['bad_coord']:4d} {st['out_of_box']:5d} "
                     f"{st['dup_name']:5d} {st['dup_coord']:5d} {st['kept']:5d} {st['in_colombo']:6d}")
    lines += ["", "-" * 70, "DROPPED ROWS", "-" * 70]
    for name, st in report.items():
        if st["dup_name_rows"] or st["dup_coord_rows"] or st["out_of_box_rows"]:
            lines.append(f"\n{name}")
            for r in st["out_of_box_rows"]:
                lines.append(f"  outside bbox    {r}")
            for r in st["dup_name_rows"]:
                lines.append(f"  duplicate name  {r}")
            for r in st["dup_coord_rows"]:
                lines.append(f"  duplicate coord {r}")
    text = "\n".join(lines) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()

"""Step 6a - one coordinate per Address, for the hazard/terrain extension.

`final_dataset.csv` carries no coordinates. Every feature in it was computed by
Step 4 from `gn_lat`/`gn_lon` in `listings_gn_clean.csv`, taking the FIRST row
seen for each division, so this table is built the same way: any other rule
would key the hazard variables to a different point than the amenity variables.

`data/reference/gn_lookup.csv` (the merged-399 coordinate file) is used as an
independent cross-check, not as the source.

Output: data/processed/gn_centroids.csv   Address, Address_ID, gn_lat, gn_lon
"""

import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "interim", "listings_gn_clean.csv")
LOOKUP = os.path.join(ROOT, "data", "reference", "gn_lookup.csv")
FINAL = os.path.join(ROOT, "data", "processed", "final_dataset.csv")
OUT = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")

# Colombo district bounds, from the ADM4 polygon set (generous by ~0.02 deg)
LAT_MIN, LAT_MAX = 6.70, 7.05
LON_MIN, LON_MAX = 79.80, 80.25


def main():
    with open(BASE, encoding="utf-8-sig") as fh:
        listings = list(csv.DictReader(fh))
    with open(FINAL, encoding="utf-8-sig") as fh:
        final = list(csv.DictReader(fh))

    # Address_ID as the built dataset assigns it - carried, not recomputed
    addr_id = {}
    for r in final:
        addr_id.setdefault(r["Address"], int(r["Address_ID"]))

    coord, variants = {}, defaultdict(set)
    for r in listings:
        a = r["gn_division"].strip().lower()
        pt = (float(r["gn_lat"]), float(r["gn_lon"]))
        coord.setdefault(a, pt)
        variants[a].add(pt)

    rows = [{"Address": a, "Address_ID": addr_id[a],
             "gn_lat": f"{coord[a][0]:.8f}", "gn_lon": f"{coord[a][1]:.8f}"}
            for a in sorted(coord, key=lambda x: addr_id[x])]

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Address", "Address_ID", "gn_lat", "gn_lon"])
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- verify
    dup_addr = [a for a, n in Counter(r["Address"] for r in rows).items() if n > 1]
    nulls = [r["Address"] for r in rows if not r["gn_lat"] or not r["gn_lon"]]
    outside = [r["Address"] for r in rows
               if not (LAT_MIN <= float(r["gn_lat"]) <= LAT_MAX
                       and LON_MIN <= float(r["gn_lon"]) <= LON_MAX)]
    multi = {a: v for a, v in variants.items() if len(v) > 1}
    final_addrs = {r["Address"] for r in final}

    with open(LOOKUP, encoding="utf-8-sig") as fh:
        lk = {r["gn_division"].strip().lower(): (float(r["lat"]), float(r["lon"]))
              for r in csv.DictReader(fh)}
    far, missing = [], []
    for r in rows:
        p = lk.get(r["Address"])
        if p is None:
            missing.append(r["Address"])
            continue
        dlat = abs(p[0] - float(r["gn_lat"]))
        dlon = abs(p[1] - float(r["gn_lon"]))
        if max(dlat, dlon) > 0.001:          # ~110 m
            far.append(f"{r['Address']} ({dlat:.4f},{dlon:.4f} deg)")

    print(f"rows written                : {len(rows)}")
    print(f"distinct Address in dataset : {len(final_addrs)}")
    print(f"covers every Address        : {final_addrs == set(r['Address'] for r in rows)}")
    print(f"duplicate Address           : {len(dup_addr)}")
    print(f"null coordinates            : {len(nulls)}")
    print(f"outside Colombo bounds      : {len(outside)} {outside}")
    print(f"divisions with >1 coordinate in the listings: {len(multi)}")
    print(f"cross-check vs gn_lookup.csv: {len(far)} disagree by >110 m, "
          f"{len(missing)} not in lookup")
    for x in far[:10]:
        print(f"    {x}")
    for x in missing[:10]:
        print(f"    missing from lookup: {x}")


if __name__ == "__main__":
    main()

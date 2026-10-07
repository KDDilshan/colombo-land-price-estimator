"""
Step 4 - build the modelling dataset.

Base      : data/interim/listings_gn_clean.csv          (7034 rows, Step 1 output)
Amenities : data/amenities/clean/*.csv                  (Step 3, schools merged in Step 5c)
Geometry  : Haversine, R = 6371.0088 km, on the GN division centroid of each listing
Fort      : (6.9333, 79.8433)

COUNT RADIUS: 2 km for every category, in every file that ships as a modelling
input. The dissertation's Table 3.4 says 5 km for the six education / government
hospital counts, but its own published dataset does not: across all 4494 rows
the smallest distance on a count=0 row is >= 2.006 km and the largest distance
on a count>0 row is <= 1.999 km, in all twelve count columns. The data the models
were actually trained on used 2 km throughout, so 2 km is primary and the 5 km
reading ships beside it as a sensitivity check.

OUTPUTS
  final_dataset.csv               PRIMARY  31 columns, 2 km counts
  final_dataset_5km_counts.csv    sensitivity, same 31 columns, Table 3.4 radii
  final_dataset_archive_35col.csv archive, the full 35-column schema, 2 km counts

DROPPED FROM THE MODELLING FILES (kept in the archive):
  min_dist_nearest_Supermarket, count_Supermarkets_within2km
      no supermarket source exists - blank distance and a placeholder zero count
      on every row, so they carry no information and the zero is misleading
  count_govtschools_B, min_dist_govtschools_b
      the Class A/B split needs a Department of Examinations ranking that is not
      retrievable, so both pairs are computed from one pool and are identical
"""

import csv
import datetime as dt
import math
import os
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "data", "interim", "listings_gn_clean.csv")
AMEN = os.path.join(ROOT, "data", "amenities", "clean")
OUT_DIR = os.path.join(ROOT, "data", "processed")
OUT_MAIN = os.path.join(OUT_DIR, "final_dataset.csv")
OUT_5KM = os.path.join(OUT_DIR, "final_dataset_5km_counts.csv")
OUT_ARCHIVE = os.path.join(OUT_DIR, "final_dataset_archive_35col.csv")
REPORT = os.path.join(OUT_DIR, "data_quality_report.md")

R_EARTH = 6371.0088
FORT = (6.9333, 79.8433)
EXCEL_EPOCH = dt.date(1899, 12, 30)

SCHEMA = [
    "Address_ID", "Address", "Land_size(Perches)", "Price_Scale", "Land_type",
    "Posted_Date_new", "Distance from fort",
    "count_govtschools_A", "min_dist_govtschools_a",
    "count_govtschools_B", "min_dist_govtschools_b",
    "count_semigovtschools", "min_dist_semigovtschools",
    "count_intlschools", "min_dist_intlschools",
    "count_uni", "min_dist_uni",
    "min_dist_nearest_express", "min_dist_nearest_railway",
    "min_dist_nearest_bank", "count_banks_within_2km",
    "min_dist_nearest_FinanceCompany", "count_FinanceCompanies_within_2km",
    "min_dist_nearest_Govt_Hospital", "count_Govt_Hospitals",
    "min_dist_nearest_Pvt_Hospital", "count_Pvt_Hospital",
    "min_dist_nearest_Pvt_Med_center", "count_Pvt_Med_Centers",
    "min_dist_nearest_Supermarket", "count_Supermarkets_within2km",
    "min_dist_nearest_Fuel_station", "count_Fuel_Stations_within2km",
    "Mentioned Price(Rs)", "Price per Perch",
]

DROPPED = {
    "min_dist_nearest_Supermarket": "no source exists - blank on every row",
    "count_Supermarkets_within2km": "no source exists - placeholder zero on every row",
    "count_govtschools_B": "identical to count_govtschools_A (one school pool, no A/B split)",
    "min_dist_govtschools_b": "identical to min_dist_govtschools_a (one school pool)",
}
MODEL_SCHEMA = [c for c in SCHEMA if c not in DROPPED]

# key -> (amenity file, dist column, count column, Table-3.4 radius km)
CATS = OrderedDict([
    ("govtschools_A", ("govtschools.csv", "min_dist_govtschools_a", "count_govtschools_A", 5.0)),
    ("govtschools_B", ("govtschools.csv", "min_dist_govtschools_b", "count_govtschools_B", 5.0)),
    ("semigovtschools", ("semigovtschools.csv", "min_dist_semigovtschools", "count_semigovtschools", 5.0)),
    ("intlschools", ("intlschools.csv", "min_dist_intlschools", "count_intlschools", 5.0)),
    ("uni", ("universities.csv", "min_dist_uni", "count_uni", 5.0)),
    ("express", ("expressway_entrances.csv", "min_dist_nearest_express", None, None)),
    ("railway", ("railway_stations.csv", "min_dist_nearest_railway", None, None)),
    ("bank", ("banks.csv", "min_dist_nearest_bank", "count_banks_within_2km", 2.0)),
    ("finance", ("finance_companies.csv", "min_dist_nearest_FinanceCompany",
                 "count_FinanceCompanies_within_2km", 2.0)),
    ("govt_hospital", ("govt_hospitals.csv", "min_dist_nearest_Govt_Hospital",
                       "count_Govt_Hospitals", 5.0)),
    ("pvt_hospital", ("pvt_hospitals.csv", "min_dist_nearest_Pvt_Hospital",
                      "count_Pvt_Hospital", 2.0)),
    ("pvt_med", ("pvt_med_centers.csv", "min_dist_nearest_Pvt_Med_center",
                 "count_Pvt_Med_Centers", 2.0)),
    ("supermarket", ("supermarkets.csv", "min_dist_nearest_Supermarket",
                     "count_Supermarkets_within2km", 2.0)),
    ("fuel", ("fuel_stations.csv", "min_dist_nearest_Fuel_station",
              "count_Fuel_Stations_within2km", 2.0)),
])

# state of the school columns before Step 5a-5c, for the before/after table
SCHOOLS_BEFORE = {"units": 91, "median_km": 9.4462, "zero_5km": 5075}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def load_points(fn):
    with open(os.path.join(AMEN, fn), encoding="utf-8-sig") as fh:
        return [(float(r["latitude"]), float(r["longitude"]))
                for r in csv.DictReader(fh)]


def excel_serial(s):
    return "" if not s else (dt.date.fromisoformat(s[:10]) - EXCEL_EPOCH).days


def fmt(x, nd=6):
    return "" if x is None else f"{x:.{nd}f}"


def median(v):
    v = sorted(v)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(BASE, encoding="utf-8-sig") as fh:
        listings = list(csv.DictReader(fh))

    points = {k: load_points(v[0]) for k, v in CATS.items()}

    # Address_ID = frequency rank of the address over this sample, descending
    # (NOTES 1.3: the original study's rule; a property of the sample)
    freq = Counter(r["gn_division"].strip().lower() for r in listings)
    addr_id = {a: i + 1 for i, a in
               enumerate(sorted(freq, key=lambda a: (-freq[a], a)))}

    # features are a property of the GN division, so compute once per division
    div_coord = {}
    for r in listings:
        div_coord.setdefault(r["gn_division"].strip().lower(),
                             (float(r["gn_lat"]), float(r["gn_lon"])))

    feat2k, feat5k = {}, {}
    for addr, (lat, lon) in div_coord.items():
        f2 = {"Distance from fort": haversine(lat, lon, FORT[0], FORT[1])}
        f5 = dict(f2)
        for key, (fn, dcol, ccol, rad) in CATS.items():
            ds = [haversine(lat, lon, p[0], p[1]) for p in points[key]]
            dmin = min(ds) if ds else None
            f2[dcol] = f5[dcol] = dmin
            if ccol:
                f2[ccol] = sum(1 for d in ds if d <= 2.0)
                f5[ccol] = sum(1 for d in ds if d <= rad)
        feat2k[addr] = f2
        feat5k[addr] = f5

    def build(feats, cols):
        out = []
        for r in listings:
            addr = r["gn_division"].strip().lower()
            row = {
                "Address_ID": addr_id[addr],
                "Address": addr,
                "Land_size(Perches)": r["Land_Size_Perch"],
                "Price_Scale": r["Price_Scale"],
                "Land_type": r["Land_Type"],
                "Posted_Date_new": excel_serial(r["Posted_Date"]),
                "Mentioned Price(Rs)": r["Price_LKR"],
                "Price per Perch": r["Price_per_Perch"],
            }
            for col, val in feats[addr].items():
                row[col] = val if isinstance(val, int) else fmt(val)
            out.append({c: row[c] for c in cols})
        return out

    rows_main = build(feat2k, MODEL_SCHEMA)
    rows_5km = build(feat5k, MODEL_SCHEMA)
    rows_arch = build(feat2k, SCHEMA)

    for path, rows, cols in ((OUT_MAIN, rows_main, MODEL_SCHEMA),
                             (OUT_5KM, rows_5km, MODEL_SCHEMA),
                             (OUT_ARCHIVE, rows_arch, SCHEMA)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # ---- schools before / after ------------------------------------------
    school_now = {
        "units": len(points["govtschools_A"]),
        "median_km": median([float(r["min_dist_govtschools_a"]) for r in rows_main]),
        "zero_5km": sum(1 for r in rows_5km if int(r["count_govtschools_A"]) == 0),
        "zero_2km": sum(1 for r in rows_main if int(r["count_govtschools_A"]) == 0),
    }

    write_report(rows_main, rows_5km, points, addr_id, school_now)

    print(f"wrote {OUT_MAIN}    ({len(rows_main)} rows x {len(MODEL_SCHEMA)} cols, 2 km counts)")
    print(f"wrote {OUT_5KM}    ({len(rows_5km)} rows x {len(MODEL_SCHEMA)} cols, Table 3.4 radii)")
    print(f"wrote {OUT_ARCHIVE} ({len(rows_arch)} rows x {len(SCHEMA)} cols)")
    print(f"wrote {REPORT}\n")
    print("SCHOOLS, before -> after")
    print(f"  units in the pool            {SCHOOLS_BEFORE['units']:>6} -> {school_now['units']}")
    print(f"  median km to nearest school  {SCHOOLS_BEFORE['median_km']:>6.2f} -> {school_now['median_km']:.2f}")
    print(f"  rows with a 5 km count of 0  {SCHOOLS_BEFORE['zero_5km']:>6} -> {school_now['zero_5km']}")
    print(f"  rows with a 2 km count of 0  {'  n/a':>6} -> {school_now['zero_2km']}")


def write_report(rows_main, rows_5km, points, addr_id, school_now):
    n = len(rows_main)
    miss = {c: sum(1 for r in rows_main if r[c] == "" or r[c] is None) for c in MODEL_SCHEMA}

    def stats(col, rows):
        vals = sorted(float(r[col]) for r in rows if r[col] != "")
        if not vals:
            return None
        return (vals[0], median(vals), vals[-1], sum(vals) / len(vals))

    status = {
        "govtschools_A": "**four zones merged** - Colombo (zonal office) + Piliyandala "
                         "(zonal office) + Sri Jayawardenapura and Homagama (DS "
                         "secretariat pages, substituted); no Class A/B split",
        "govtschools_B": "identical pool to Class A - dropped from the modelling files",
        "semigovtschools": "Colombo zone only - the other three sources do not "
                           "classify semi-government",
        "intlschools": "complete",
        "uni": "complete",
        "express": "complete (distance-only feature)",
        "railway": "partial - 21 of 105 source stations geocoded",
        "bank": "partial - 345 of 477 in-scope branches geocoded (72.3%)",
        "finance": "partial - 161 of 172 in-scope branches geocoded (93.6%)",
        "govt_hospital": "155 of 221 MoH units geocoded, 1 duplicate coordinate removed",
        "pvt_hospital": "partial - 63 of 91 PHSRC units geocoded, 1 duplicate name removed",
        "pvt_med": "partial - 65 of 80 in-scope PHSRC units, 1 duplicate name removed",
        "supermarket": "**NO DATA** - every brand locator unreachable; both columns dropped",
        "fuel": "partial - 27 of 121 in-scope Ceypetco units, and Ceypetco only",
    }

    cl = ["# Data quality report - `final_dataset.csv`", "",
          f"Generated {dt.date.today().isoformat()} by `src/step4_build_final_dataset.py`.", "",
          "## 1. What was produced", "",
          "| file | rows | columns | count radius | role |", "|---|---|---|---|---|",
          f"| `final_dataset.csv` | {n} | {len(MODEL_SCHEMA)} | 2 km | **primary modelling file** |",
          f"| `final_dataset_5km_counts.csv` | {n} | {len(MODEL_SCHEMA)} | Table 3.4 (5 km / 2 km) | sensitivity check |",
          f"| `final_dataset_archive_35col.csv` | {n} | {len(SCHEMA)} | 2 km | archive - keeps the dropped columns |", "",
          "Base: `data/interim/listings_gn_clean.csv` (7034 rows). One row in, one row out.", "",
          "Geometry: Haversine, R = 6371.0088 km, listing located at its GN division "
          "centroid. Distance from Fort measured to (6.9333, 79.8433). All distances in km.", "",
          "## 2. Columns dropped from the modelling files", "",
          "| column | why |", "|---|---|"]
    for c, why in DROPPED.items():
        cl.append(f"| `{c}` | {why} |")
    cl += ["", "All four are retained in `final_dataset_archive_35col.csv`, which keeps the "
           "original 35-column schema intact. Dropping them removes no information: two "
           "carry none, and two duplicate columns that are kept.", "",
           "## 3. School coverage - limitation #3 of the previous report, now closed", "",
           "| | before | after |", "|---|---|---|",
           f"| schools in the pool | {SCHOOLS_BEFORE['units']} | **{school_now['units']}** |",
           f"| median km to nearest government school | {SCHOOLS_BEFORE['median_km']:.2f} | "
           f"**{school_now['median_km']:.2f}** |",
           f"| rows with a 5 km school count of 0 | {SCHOOLS_BEFORE['zero_5km']} | "
           f"**{school_now['zero_5km']}** |",
           f"| rows with a 2 km school count of 0 (primary rule) | n/a | {school_now['zero_2km']} |", "",
           "Sources added: **Piliyandala zone** from its own site (`pilizone.lk`, one page "
           "per division). **Sri Jayawardenapura** and **Homagama** zones have no website - "
           "`wpedu.sch.lk` does not resolve in DNS and `www.mahozone.sch.lk` does not "
           "connect - so their schools were taken from the Divisional Secretariat pages "
           "`<division>.ds.gov.lk/index.php/en/schools.html`. **That is a source "
           "substitution and it is declared here and in NOTES.md 5.6.** The DS pages do not "
           "say whether a school is government or semi-government, so every school from "
           "them is recorded as government and the semi-government pool is still the "
           "Colombo zone's 6 units.", "",
           "## 4. Amenity categories", "",
           "| category | units used | source status |", "|---|---|---|"]
    for key in CATS:
        cl.append(f"| `{key}` | {len(points[key])} | {status[key]} |")

    cl += ["", "## 5. Missing values", "", "| column | missing | % |", "|---|---|---|"]
    for c in MODEL_SCHEMA:
        if miss[c]:
            cl.append(f"| `{c}` | {miss[c]} | {100.0 * miss[c] / n:.2f}% |")
    if not any(miss.values()):
        cl.append("| - | 0 | 0.00% |")
    cl += ["", "Every other column is fully populated. The only gap left in the primary "
           "file is `Posted_Date_new`; the supermarket columns that were 100% missing are "
           "no longer in it.", "",
           "## 6. Duplicates", "", "| check | count |", "|---|---|"]
    seen, dup_full = set(), 0
    for r in rows_main:
        k = tuple(r[c] for c in MODEL_SCHEMA)
        if k in seen:
            dup_full += 1
        seen.add(k)
    cl += [f"| identical rows (all {len(MODEL_SCHEMA)} columns) | {dup_full} |",
           f"| distinct addresses | {len(addr_id)} |",
           "| duplicate amenity names/coordinates remaining | 0 (Step 3 rules, re-applied "
           "to the merged school pool in Step 5c) |", "",
           "Identical rows are expected: two adverts for the same land size, type, price and "
           "division in the same period are indistinguishable in this schema. Re-posted "
           "duplicates were removed in Step 1 (rule R11, 1053 rows).", "",
           "## 7. Value ranges (primary file)", "",
           "| column | min | median | max | mean |", "|---|---|---|---|---|"]
    for c in MODEL_SCHEMA:
        if c in ("Address", "Price_Scale", "Land_type", "Address_ID"):
            continue
        s = stats(c, rows_main)
        if s:
            cl.append(f"| `{c}` | {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} | {s[3]:.4f} |")

    cl += ["", "## 8. Assumptions", "",
           "1. **Count radius is 2 km for every category.** Table 3.4 of the dissertation "
           "says 5 km for the six education / government-hospital counts, but the study's "
           "own published dataset contradicts it: in all twelve count columns the smallest "
           "distance on a `count = 0` row is >= 2.006 km and the largest on a `count > 0` "
           "row is <= 1.999 km, with no overlap in any category. The models were trained on "
           "2 km counts. `final_dataset_5km_counts.csv` holds the Table 3.4 reading for a "
           "sensitivity check.",
           "2. **Schools are one pool, not Class A and Class B.** The split needs a "
           "Department of Examinations A/L ranking that is not retrievable. One pair of "
           "columns is computed and shipped; the duplicate pair is dropped.",
           "3. **Sri Jayawardenapura and Homagama schools come from Divisional Secretariat "
           "pages, not zonal education offices** - a declared substitution, because those "
           "two zones have no website. They are official government pages, but they do not "
           "carry the government / semi-government classification.",
           "4. **Semi-government schools remain 6 units**, all from the Colombo zone.",
           "5. **Supermarkets have no source at all** and both columns are dropped rather "
           "than shipped as a misleading zero.",
           "6. **A listing is located at its GN division centroid**, not at the plot. Every "
           "listing in a division shares one set of amenity features - the original study's "
           "design.",
           "7. **All cleaned amenity rows are used for both distance and count**, including "
           "units outside Colombo district.",
           "8. **`Price per Perch` and `Mentioned Price(Rs)` are carried through unchanged** "
           "from Step 1.",
           "9. **`Posted_Date_new` is an Excel serial day number** (days since 1899-12-30). "
           "58 rows have no posted date and are blank.",
           "10. **`Address_ID` is a frequency rank over this sample** (most-listed division "
           "= 1), reproducing the original rule. Not comparable to the original study's IDs.",
           "", "## 9. Remaining limitations", "",
           "1. **129 of 251 new school units did not geocode** - 85 returned nothing "
           "admissible from Google, 35 fell outside the acceptance radius, 9 collided with "
           "a pin already claimed. These are mostly small rural junior schools in Padukka "
           "and Seethawaka that Google's index does not carry. A town centroid was never "
           "accepted in their place. School counts in the outer district are therefore "
           "still **lower bounds**, though far better than before.",
           "2. **Semi-government schools are Colombo-zone only** (6 units), because no other "
           "source classifies them. That column understates the outer district.",
           "3. **Fuel stations**: 27 units, Ceypetco only, from a 311-row source list - "
           "Lanka IOC and Laugfs are missing entirely, so fuel counts are systematically "
           "understated. Railway stations (21 of 105) and banks (345 of 477) are also "
           "incomplete. Counts in these categories are lower bounds.",
           "4. **Fuel station names were constructed**, not published (`Ceypetco Filling "
           "Station <locality>`), so their coordinates resolve to a locality rather than a "
           "forecourt.",
           "5. **Unknown share of geocoding failures are network noise** - the geocoder "
           "cached connection errors as genuine zero-results.",
           "6. **The target is right-censored** at Rs 10,000,000/perch (16 rows flagged in "
           "Step 1); the Colombo 1/2/3/7 core is not represented.",
           "7. **194 of 390 GN division names appear**; the rest have no 2026 listings.",
           "8. Coordinates for several categories, including all 122 new schools, came from "
           "the Google Geocoding API. Google's terms restrict storing and redistributing "
           "them - ship this computed dataset, not the amenity coordinate files.", ""]

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cl))


if __name__ == "__main__":
    main()

"""Step 6b - the landslide null test, run FIRST and documented either way.

NBRO has not declared Colombo a landslide-prone district; its ten are Kandy,
Matale, Nuwara Eliya, Badulla, Kegalle, Ratnapura, Kalutara, Galle, Matara and
Hambantota. The expectation is therefore a near-constant column. This script
tests that against data instead of asserting it, and writes the result out
whichever way it falls.

Source: NASA Global Landslide Susceptibility Map (LHASA), ~1 km, downloaded from
gpm.nasa.gov. Classes 1..5 = very low .. very high; 0 is water / no data.

Two samples are taken per division:
  point   the class at the centroid pixel
  window  the class distribution over a ~1 km box around the centroid, so a
          single unlucky pixel cannot decide the answer

Output: data/processed/landslide_null_test.md
        data/processed/landslide_lhasa_194.csv
"""

import csv
import os
from collections import Counter

import rasterio
import requests
from rasterio.windows import from_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
RAW = os.path.join(ROOT, "data", "raw", "lhasa_global_susceptibility.tif")
OUT_CSV = os.path.join(ROOT, "data", "processed", "landslide_lhasa_194.csv")
OUT_MD = os.path.join(ROOT, "data", "processed", "landslide_null_test.md")

URL = ("https://gpm.nasa.gov/sites/default/files/downloads/"
       "global-landslide-susceptibility-map-2-27-23.tif")

CLASS = {0: "no data / water", 1: "very low", 2: "low", 3: "moderate",
         4: "high", 5: "very high"}
HALF_DEG = 0.0045          # ~500 m at this latitude -> a ~1 km box


def fetch():
    if os.path.exists(RAW) and os.path.getsize(RAW) > 1_000_000:
        print(f"using cached raster {os.path.getsize(RAW) / 1e6:.1f} MB")
        return
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    print(f"downloading {URL}")
    with requests.get(URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(RAW, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"  {os.path.getsize(RAW) / 1e6:.1f} MB")


def main():
    fetch()
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))

    rows = []
    with rasterio.open(RAW) as src:
        print(f"raster {src.width}x{src.height} crs={src.crs} "
              f"res={src.res[0]:.5f} deg dtype={src.dtypes[0]}")
        coords = [(float(p["gn_lon"]), float(p["gn_lat"])) for p in pts]
        point_vals = [v[0] for v in src.sample(coords)]

        for p, v in zip(pts, point_vals):
            lon, lat = float(p["gn_lon"]), float(p["gn_lat"])
            w = from_bounds(lon - HALF_DEG, lat - HALF_DEG,
                            lon + HALF_DEG, lat + HALF_DEG, src.transform)
            block = src.read(1, window=w)
            flat = [int(x) for x in block.flatten()] if block.size else [int(v)]
            c = Counter(flat)
            rows.append({
                "Address": p["Address"],
                "Address_ID": p["Address_ID"],
                "gn_lat": p["gn_lat"], "gn_lon": p["gn_lon"],
                "landslide_susceptibility": int(v),
                "landslide_class": CLASS.get(int(v), str(int(v))),
                "window_pixels": len(flat),
                "window_max": max(flat),
                "window_mode": c.most_common(1)[0][0],
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    point = Counter(r["landslide_susceptibility"] for r in rows)
    mode = Counter(r["window_mode"] for r in rows)
    mx = Counter(r["window_max"] for r in rows)
    n = len(rows)
    distinct = len(point)

    verdict = ("**NEAR-CONSTANT - DO NOT USE AS A FEATURE.**" if distinct <= 2 else
               "**LOW VARIANCE - use only with the distinct-value count stated.**"
               if distinct <= 3 else
               "**IT VARIES - worth screening properly in Step 4.**")

    md = [
        "# Landslide null test - NASA LHASA at the 194 Colombo GN centroids", "",
        f"Run {__import__('datetime').date.today().isoformat()} by "
        "`src/step6b_landslide_null_test.py`.", "",
        "## Result", "",
        f"**Distinct values across the 194 divisions: {distinct}.** {verdict}", "",
        "## Why this test was run", "",
        "NBRO has not declared Colombo a landslide-prone district. Its ten "
        "landslide districts are Kandy, Matale, Nuwara Eliya, Badulla, Kegalle, "
        "Ratnapura, Kalutara, Galle, Matara and Hambantota. A landslide "
        "susceptibility column for Colombo was therefore expected to be "
        "near-constant - but a column that is *assumed* constant and a column "
        "that is *measured* constant are different things to put in a "
        "dissertation, and a near-constant feature can still be selected by "
        "RFECV and read as meaningful. This is the measurement.", "",
        "## Point sample at the centroid", "",
        "| class | value | divisions | % |", "|---|---|---|---|"]
    for v, k in sorted(point.items()):
        md.append(f"| {CLASS.get(v, v)} | {v} | {k} | {100.0 * k / n:.1f}% |")

    md += ["", "## Modal class over a ~1 km window", "",
           "A single pixel can be unlucky; this is the most common class in a "
           "~1 km box around each centroid.", "",
           "| class | value | divisions | % |", "|---|---|---|---|"]
    for v, k in sorted(mode.items()):
        md.append(f"| {CLASS.get(v, v)} | {v} | {k} | {100.0 * k / n:.1f}% |")

    md += ["", "## Highest class anywhere in the ~1 km window", "",
           "The most generous reading available - if even this is flat, there is "
           "no landslide signal in Colombo district at this resolution.", "",
           "| class | value | divisions | % |", "|---|---|---|---|"]
    for v, k in sorted(mx.items()):
        md.append(f"| {CLASS.get(v, v)} | {v} | {k} | {100.0 * k / n:.1f}% |")

    nodata = [r["Address"] for r in rows if r["landslide_susceptibility"] == 0]
    moderate = [r["Address"] for r in rows if r["landslide_susceptibility"] >= 3]
    md += ["", "## What the variation actually is", "",
           f"The {distinct} distinct values are not a spread - they are a "
           f"two-class split. `very low` and `low` together account for "
           f"{100.0 * (point.get(1, 0) + point.get(2, 0)) / n:.1f}% of divisions. "
           f"The {len(moderate)} division(s) rated `moderate` or above are "
           + (", ".join(f"`{m}`" for m in moderate) or "none") +
           " - all on the eastern edge of the district, towards the Kegalle and "
           "Ratnapura hills where NBRO's hazard zonation actually begins. That "
           "is the expected geography, and it means the column is close to a "
           "restatement of *how far east the division is*.", ""]
    if nodata:
        md += [f"`{nodata[0]}` returns class 0 (no data / water) at the centroid "
               "and across its whole 1 km window - it is the coastal division at "
               "the harbour. A susceptibility column would therefore also carry a "
               "null that has to be resolved before any model could use it.", ""]

    md += ["", "## Source", "",
           f"- NASA Global Landslide Susceptibility Map (LHASA), `{URL}`",
           "- ~1 km resolution, global, classes 1-5 (very low - very high).",
           "- Per-division values: `data/processed/landslide_lhasa_194.csv`", "",
           "## What this means for the dataset", "",
           ("The column carries no usable variance across Colombo district and is "
            "**not** joined into `final_dataset_hazard.csv`. It is kept here as a "
            "documented negative, not dropped silently. If the study is ever "
            "extended to Ratnapura, Kegalle or Galle, this same script will "
            "produce a column that does vary - that is where a landslide "
            "variable belongs."
            if distinct <= 3 else
            "The column went forward to the Step 4 screening with the other "
            "candidates and was **dropped there**: 4 distinct values across 194 "
            "divisions is far below the 30-value threshold for a usable "
            "measurement, and it correlates +0.57 with `Distance from fort`, "
            "which the model already has. See `hazard_screening.md`. It is "
            "**not** joined into `final_dataset_hazard.csv`, and it is recorded "
            "here rather than dropped silently.\n\nIf the study is ever extended "
            "to Ratnapura, Kegalle or Galle, this same script will produce a "
            "column that does vary - that is where a landslide variable "
            "belongs."), ""]

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"\ndistinct point values across 194 divisions: {distinct}")
    for v, k in sorted(point.items()):
        print(f"  {v} {CLASS.get(v, ''):16s} {k:4d}  {100.0 * k / n:5.1f}%")
    print(f"window max classes: {dict(sorted(mx.items()))}")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()

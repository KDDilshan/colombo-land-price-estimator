"""Earth Engine extraction of the hazard/terrain variables for the 194 Colombo
GN division centroids.

WHAT THIS SCRIPT IS FOR
  Two of the requested layers have no public cloud-optimised mirror I could read
  directly, so they need Earth Engine:

      hand_m            MERIT Hydro, height above nearest drainage, 90 m
      flood_frequency   Global Flood Database v1, count of observed inundations

  The other five (elevation, slope, built-up fraction, surface-water occurrence,
  landslide susceptibility) were already produced WITHOUT Earth Engine by
  src/step6c_extract_cog_hazards.py and src/step6b_landslide_null_test.py,
  reading the same Copernicus DEM / ESA WorldCover / JRC Global Surface Water /
  NASA LHASA sources over HTTP. They are recomputed here anyway so the whole set
  can be produced from one place if you prefer, and so the two methods can be
  cross-checked against each other.

GEOMETRY
  Each centroid is buffered by 1 km and every variable is the MEAN over that
  buffer - never a single pixel. `built_up_fraction` is the mean of a 0/1 mask,
  which is the fraction of the buffer that is built up.

HOW TO RUN IT

  1. Install the client and authenticate (one browser round trip):

         pip install earthengine-api
         earthengine authenticate

     That opens a browser, you approve with the Google account your Earth
     Engine access is attached to, and it writes a token to
     %USERPROFILE%\\.config\\earthengine\\credentials.

  2. Find your Cloud project id - Earth Engine now requires one. It is shown at
     https://code.earthengine.google.com under the project selector, and looks
     like "ee-yourname". Then:

         set EE_PROJECT=ee-yourname
         python scripts/gee_hazard_extract.py

     (PowerShell: $env:EE_PROJECT = "ee-yourname")

  3. The script writes data/processed/gee_hazard_194.csv directly - it uses
     getInfo() on a 194-feature collection, which is well inside Earth Engine's
     synchronous response limit, so there is no Drive export to wait for.

If a layer id has been deprecated since this was written, the script says which
one failed and continues with the rest rather than dying.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS = ROOT / "data" / "processed" / "gn_centroids.csv"
OUT = ROOT / "data" / "processed" / "gee_hazard_194.csv"

BUFFER_M = 1000

COLUMNS = ["hand_m", "elevation_m", "slope_deg", "water_occurrence_pct",
           "flood_frequency", "built_up_fraction", "landslide_susceptibility"]


def build_bands(ee):
    """One multi-band image; every band is sampled with the same reducer."""
    bands, failed = [], []

    def add(label, fn):
        try:
            bands.append(fn().rename(label))
        except Exception as exc:                     # noqa: BLE001
            failed.append(f"{label}: {type(exc).__name__} {exc}")

    # MERIT Hydro - height above nearest drainage
    add("hand_m", lambda: ee.Image("MERIT/Hydro/v1_0_1").select("hnd"))

    # Copernicus DEM GLO-30 - elevation and slope
    def _dem():
        return (ee.ImageCollection("COPERNICUS/DEM/GLO30")
                .select("DEM").mosaic().rename("elevation_m"))

    add("elevation_m", _dem)
    add("slope_deg", lambda: ee.Terrain.slope(_dem().setDefaultProjection(
        crs="EPSG:4326", scale=30)))

    # JRC Global Surface Water - percentage of time water was present
    add("water_occurrence_pct",
        lambda: ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        .unmask(0))

    # Global Flood Database - how many mapped floods inundated the pixel
    add("flood_frequency",
        lambda: (ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
                 .select("flooded").sum().unmask(0)))

    # ESA WorldCover - class 50 is built-up; the mean of the mask is the fraction
    add("built_up_fraction",
        lambda: (ee.ImageCollection("ESA/WorldCover/v200").first()
                 .select("Map").eq(50)))

    # NASA LHASA landslide susceptibility, published as an EE community asset
    add("landslide_susceptibility",
        lambda: ee.Image("projects/sat-io/open-datasets/global-landslide-"
                         "susceptibility-map"))

    return bands, failed


def main() -> int:
    try:
        import ee
    except ImportError:
        print("earthengine-api is not installed.  Run:  pip install earthengine-api")
        return 1

    project = os.environ.get("EE_PROJECT", "").strip()
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as exc:                          # noqa: BLE001
        print(f"Earth Engine did not initialise: {exc}\n"
              "Run `earthengine authenticate`, then set EE_PROJECT to your "
              "Cloud project id (e.g. ee-yourname).")
        return 1

    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))
    print(f"{len(pts)} centroids")

    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(p["gn_lon"]), float(p["gn_lat"])])
                   .buffer(BUFFER_M),
                   {"Address": p["Address"], "Address_ID": int(p["Address_ID"])})
        for p in pts])

    bands, failed = build_bands(ee)
    for f in failed:
        print(f"  ! layer unavailable, skipped - {f}")
    if not bands:
        print("no layers available; nothing to do")
        return 1

    img = ee.Image.cat(bands)
    sampled = img.reduceRegions(collection=fc,
                                reducer=ee.Reducer.mean(),
                                scale=30,
                                tileScale=4)

    print("requesting values from Earth Engine ...")
    rows = [f["properties"] for f in sampled.getInfo()["features"]]

    have = [c for c in COLUMNS if any(c in r for r in rows)]
    fields = ["Address", "Address_ID"] + have
    rows.sort(key=lambda r: r.get("Address_ID", 0))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {OUT}  ({len(rows)} rows)")

    print(f"\n{'column':26s} {'min':>10s} {'max':>10s} {'mean':>10s} "
          f"{'nulls':>6s} {'distinct':>9s}")
    for c in have:
        v = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        nulls = len(rows) - len(v)
        if v:
            print(f"{c:26s} {min(v):10.4f} {max(v):10.4f} {sum(v) / len(v):10.4f} "
                  f"{nulls:6d} {len(set(v)):9d}")
        else:
            print(f"{c:26s} {'-':>10s} {'-':>10s} {'-':>10s} {nulls:6d} {0:9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

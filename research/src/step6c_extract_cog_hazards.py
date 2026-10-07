"""Step 6c - terrain and surface-water variables from public cloud-optimised
GeoTIFFs, read directly over HTTP.

WHY NOT EARTH ENGINE FOR THESE.  Earth Engine needs an interactive OAuth login I
cannot perform. Three of the requested layers are published as public COGs and
can be read with windowed HTTP range requests instead, from the SAME sources
the brief names - Copernicus DEM, ESA WorldCover, JRC Global Surface Water. No
substitution is involved. The two layers with no public COG mirror
(MERIT Hydro HAND, Global Flood Database) stay in scripts/gee_hazard_extract.py
for the user to run.

GEOMETRY.  Every variable is the mean over a 1 km-radius circular buffer around
the division centroid - never a single pixel. The box read is +/-1 km and pixels
outside the circle are masked out.

  elevation_m           Copernicus DEM GLO-30, 30 m, mean over the buffer
  slope_deg             derived from the same DEM window, mean over the buffer
  built_up_fraction     ESA WorldCover 2021 v200, 10 m, share of class 50
  tree_cover_fraction   same layer, share of class 10 (free, and the NDVI-style
                        greenness proxy the brief's earlier discussion wanted)
  water_occurrence_pct  JRC Global Surface Water 2021, 30 m, mean occurrence

Output: data/processed/cog_hazard_194.csv
"""

import csv
import math
import os
from collections import OrderedDict

import numpy as np
import rasterio
from rasterio.windows import from_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
OUT = os.path.join(ROOT, "data", "processed", "cog_hazard_194.csv")

BUFFER_KM = 1.0
DEG_PER_KM_LAT = 1.0 / 110.574

DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
       "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/"
              "map/ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif")
GSW = ("https://storage.googleapis.com/global-surface-water/downloads2021/"
       "occurrence/occurrence_{lon:d}E_10Nv1_4_2021.tif")

WC_BUILTUP, WC_TREE = 50, 10
GSW_NODATA = 255

_open = {}


def src(url):
    if url not in _open:
        _open[url] = rasterio.open(url)
    return _open[url]


def deg_box(lat, lon, km):
    dlat = km * DEG_PER_KM_LAT
    dlon = km * DEG_PER_KM_LAT / max(math.cos(math.radians(lat)), 1e-6)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def circle_mask(win_transform, shape, lat, lon, km):
    """True where the pixel centre is within `km` of (lat, lon)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    xs = win_transform.c + (cols + 0.5) * win_transform.a
    ys = win_transform.f + (rows + 0.5) * win_transform.e
    dy = (ys - lat) / DEG_PER_KM_LAT
    dx = (xs - lon) * math.cos(math.radians(lat)) / DEG_PER_KM_LAT
    return (dx * dx + dy * dy) <= km * km


def read_circle(url, lat, lon, km):
    """Values inside the buffer, plus the window and its transform."""
    s = src(url)
    w = from_bounds(*deg_box(lat, lon, km), s.transform)
    arr = s.read(1, window=w, boundless=True, fill_value=0)
    if arr.size == 0:
        return None, None, None
    t = s.window_transform(w)
    m = circle_mask(t, arr.shape, lat, lon, km)
    return arr, m, t


def dem_tiles(lat, lon, km):
    x0, y0, x1, y1 = deg_box(lat, lon, km)
    return {(int(math.floor(y)), int(math.floor(x)))
            for y in (y0, y1) for x in (x0, x1)}


def gsw_tiles(lat, lon, km):
    x0, _, x1, _ = deg_box(lat, lon, km)
    return {int(math.floor(x / 10.0) * 10) for x in (x0, x1)}


def wc_tile(lat, lon):
    return (int(math.floor(lat / 3.0) * 3), int(math.floor(lon / 3.0) * 3))


def slope_from(dem, mask, lat):
    """Mean slope in degrees over the buffer, from the DEM window."""
    if dem is None or dem.shape[0] < 3 or dem.shape[1] < 3:
        return None
    px_y = 30.0
    px_x = 30.0 * math.cos(math.radians(lat)) / math.cos(math.radians(lat))
    gy, gx = np.gradient(dem.astype("float64"), px_y, px_x)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    v = slope[mask]
    return float(np.mean(v)) if v.size else None


def main():
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))

    rows = []
    for i, p in enumerate(pts, 1):
        lat, lon = float(p["gn_lat"]), float(p["gn_lon"])
        rec = OrderedDict([("Address", p["Address"]), ("Address_ID", p["Address_ID"]),
                           ("gn_lat", p["gn_lat"]), ("gn_lon", p["gn_lon"])])

        # ---- Copernicus DEM: elevation + slope ---------------------------
        vals, primary = [], None
        for (tlat, tlon) in sorted(dem_tiles(lat, lon, BUFFER_KM)):
            url = DEM.format(lat=tlat, lon=tlon)
            try:
                arr, m, t = read_circle(url, lat, lon, BUFFER_KM)
            except Exception:
                continue
            if arr is None:
                continue
            good = m & (arr > -1000) & (arr != 0)
            vals.extend(arr[good].tolist())
            if (tlat, tlon) == (int(math.floor(lat)), int(math.floor(lon))):
                primary = (arr.astype("float64"), good)
        rec["elevation_m"] = round(float(np.mean(vals)), 3) if vals else ""
        rec["slope_deg"] = ("" if primary is None
                            else round(slope_from(primary[0], primary[1], lat) or 0.0, 4))

        # ---- ESA WorldCover: built-up and tree fractions ------------------
        tlat, tlon = wc_tile(lat, lon)
        try:
            arr, m, _ = read_circle(WORLDCOVER.format(lat=tlat, lon=tlon),
                                    lat, lon, BUFFER_KM)
            v = arr[m & (arr > 0)]
            rec["built_up_fraction"] = (round(float(np.mean(v == WC_BUILTUP)), 5)
                                        if v.size else "")
            rec["tree_cover_fraction"] = (round(float(np.mean(v == WC_TREE)), 5)
                                          if v.size else "")
            rec["worldcover_pixels"] = int(v.size)
        except Exception as exc:
            print(f"  ! worldcover {p['Address']}: {type(exc).__name__}")
            rec["built_up_fraction"] = rec["tree_cover_fraction"] = ""
            rec["worldcover_pixels"] = 0

        # ---- JRC Global Surface Water occurrence --------------------------
        vals = []
        for tlon in sorted(gsw_tiles(lat, lon, BUFFER_KM)):
            try:
                arr, m, _ = read_circle(GSW.format(lon=tlon), lat, lon, BUFFER_KM)
            except Exception:
                continue
            if arr is None:
                continue
            good = m & (arr != GSW_NODATA)
            vals.extend(arr[good].tolist())
        rec["water_occurrence_pct"] = round(float(np.mean(vals)), 4) if vals else ""
        rows.append(rec)
        if i % 20 == 0 or i == len(pts):
            print(f"  [{i:3d}/{len(pts)}] {p['Address']}")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT}\n")
    print(f"{'column':24s} {'min':>10s} {'max':>10s} {'mean':>10s} {'nulls':>6s} {'distinct':>9s}")
    for c in ("elevation_m", "slope_deg", "built_up_fraction",
              "tree_cover_fraction", "water_occurrence_pct"):
        v = [float(r[c]) for r in rows if r[c] != ""]
        nulls = sum(1 for r in rows if r[c] == "")
        print(f"{c:24s} {min(v):10.4f} {max(v):10.4f} {sum(v) / len(v):10.4f} "
              f"{nulls:6d} {len(set(v)):9d}")


if __name__ == "__main__":
    main()

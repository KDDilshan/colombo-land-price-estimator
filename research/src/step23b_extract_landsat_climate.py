"""Step 23b - day LST, NDVI and NDBI for the 194 divisions, read from the
public Landsat Collection 2 Level-2 COGs on Microsoft Planetary Computer.

WHY THIS EXISTS.  scripts/gee_climate_vars.js (step23a) is the extraction of
record, but Earth Engine needs an interactive login that cannot be performed
here - the same constraint step6c records. Planetary Computer hosts the
identical USGS product (Landsat 8/9 C2 T1 L2) as anonymously readable COGs, so
this script reproduces the GEE pipeline step for step and the downstream work
is not blocked. step24 uses the GEE export when present and reports agreement
between the two.

MIRRORS step23a EXACTLY
  scenes      landsat-8 + landsat-9, collection category T1,
              2023-01-01 .. 2025-12-31, landsat:cloud_cover_land < 40
  mask        QA_PIXEL bits 0-4 clear and ST_B10 > 0
  scaling     SR x 0.0000275 - 0.2 ; ST x 0.00341802 + 149.0 - 273.15
  indices     per scene, then per-pixel median over clear observations
  reduction   mean over the pixels whose centres fall inside the 1 km circle
              around the division point (step6c's rule), EPSG:32644, 30 m

MEMORY.  The per-scene index values for every pixel inside the union of the
194 buffers are written to int16 memmaps on disk (x10000 for the indices,
x100 for LST) and the median is taken in pixel chunks, so peak RAM stays small.
The run is resumable: finished scenes are recorded and skipped on re-run.

Output: data/processed/landsat_climate_194.csv
        Address, lst_day_c, ndvi, ndbi, water_share, clear_obs, pixels
"""

import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform

for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
             "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
             "GDAL_HTTP_MULTIRANGE": "YES",
             "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
             "GDAL_HTTP_MAX_RETRY": "6",
             "GDAL_HTTP_RETRY_DELAY": "3",
             "VSI_CACHE": "TRUE"}.items():
    os.environ.setdefault(k, v)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
OUT = os.path.join(ROOT, "data", "processed", "landsat_climate_194.csv")
CACHE = os.environ.get("LANDSAT_CACHE",
                       os.path.join(ROOT, "data", "interim", "_landsat_cache"))

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
START, END = "2023-01-01", "2025-12-31"
CLOUD_LAND_MAX = 40
CRS = "EPSG:32644"
RES = 30.0
BUFFER_M = 1000.0
NODATA = -32768
BANDS = ["red", "nir08", "swir16", "lwir11", "qa_pixel"]
CLEAR_BITS = 0b11111            # bits 0-4: fill, dilated cloud, cirrus, cloud, shadow
WATER_BIT = 1 << 7


def load_points():
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))
    lon = [float(p["gn_lon"]) for p in pts]
    lat = [float(p["gn_lat"]) for p in pts]
    xs, ys = warp_transform("EPSG:4326", CRS, lon, lat)
    return [p["Address"] for p in pts], np.array(xs), np.array(ys)


def grid(xs, ys):
    """Target grid on the Landsat 30 m lattice (pixel edges at 15 m + k*30)."""
    pad = BUFFER_M + 2 * RES
    x0 = math.floor((xs.min() - pad - 15) / RES) * RES + 15
    y1 = math.ceil((ys.max() + pad - 15) / RES) * RES + 15
    w = int(math.ceil((xs.max() + pad - x0) / RES))
    h = int(math.ceil((y1 - (ys.min() - pad)) / RES))
    return from_origin(x0, y1, RES, RES), w, h


def buffer_index(tr, w, h, xs, ys):
    """Flat pixel indices inside each 1 km circle, and their union."""
    cx = tr.c + (np.arange(w) + 0.5) * RES
    cy = tr.f - (np.arange(h) + 0.5) * RES
    per = []
    for x, y in zip(xs, ys):
        c0, c1 = np.searchsorted(cx, x - BUFFER_M), np.searchsorted(cx, x + BUFFER_M)
        r0 = np.searchsorted(-cy, -(y + BUFFER_M))
        r1 = np.searchsorted(-cy, -(y - BUFFER_M))
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        inside = (cx[cc] - x) ** 2 + (cy[rr] - y) ** 2 <= BUFFER_M ** 2
        per.append((rr[inside] * w + cc[inside]).astype(np.int64))
    union = np.unique(np.concatenate(per))
    return per, union


def search():
    cat = pystac_client.Client.open(STAC)
    items = cat.search(collections=["landsat-c2-l2"],
                       bbox=[79.80, 6.70, 80.25, 7.05],
                       datetime=f"{START}/{END}").item_collection()
    keep = [i for i in items
            if i.properties.get("platform") in ("landsat-8", "landsat-9")
            and i.properties.get("landsat:collection_category") == "T1"
            and i.properties.get("landsat:cloud_cover_land", 100) < CLOUD_LAND_MAX]
    keep.sort(key=lambda i: i.id)
    return keep


def read_band(href, tr, w, h):
    for attempt in range(4):
        try:
            with rasterio.open(pc.sign(href)) as src:
                with WarpedVRT(src, crs=CRS, transform=tr, width=w, height=h,
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    return vrt.read(1)
        except Exception as e:                     # flaky mobile link
            if attempt == 3:
                raise
            print(f"      retry {attempt + 1}: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
            time.sleep(5 * (attempt + 1))


def scene_values(item, tr, w, h, union):
    with ThreadPoolExecutor(max_workers=5) as ex:
        arrs = dict(zip(BANDS, ex.map(
            lambda b: read_band(item.assets[b].href, tr, w, h).ravel()[union],
            BANDS)))
    qa, st = arrs["qa_pixel"], arrs["lwir11"]
    clear = ((qa & CLEAR_BITS) == 0) & (qa != 0) & (st > 0)
    red, nir, swir = (arrs[b].astype(np.float32) * 0.0000275 - 0.2
                      for b in ("red", "nir08", "swir16"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndbi = (swir - nir) / (swir + nir)
    lst = st.astype(np.float32) * 0.00341802 + 149.0 - 273.15
    ok = clear & np.isfinite(ndvi) & np.isfinite(ndbi)

    def q(a, s):
        out = np.full(a.shape, NODATA, np.int16)
        out[ok] = np.clip(np.round(a[ok] * s), -32767, 32767).astype(np.int16)
        return out
    water = np.where(ok, ((qa & WATER_BIT) != 0).astype(np.int8), -1)
    return q(ndvi, 10000), q(ndbi, 10000), q(lst, 100), water, int(ok.sum())


def main():
    os.makedirs(CACHE, exist_ok=True)
    names, xs, ys = load_points()
    tr, w, h = grid(xs, ys)
    per, union = buffer_index(tr, w, h, xs, ys)
    items = search()
    n, p = len(items), len(union)
    print(f"grid {w} x {h} px; union of buffers {p:,} px; scenes {n}", flush=True)

    shape = (n, p)
    meta_path = os.path.join(CACHE, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("ids") != [i.id for i in items] or meta.get("pixels") != p:
            print("cache belongs to a different scene list - starting over")
            meta = {}
    mode = "r+" if meta else "w+"
    mm = {k: np.lib.format.open_memmap(os.path.join(CACHE, f"{k}.npy"), mode=mode,
                                       dtype=np.int8 if k == "water" else np.int16,
                                       shape=shape)
          for k in ("ndvi", "ndbi", "lst", "water")}
    if not meta:
        meta = {"ids": [i.id for i in items], "pixels": p, "done": {}}
    done = meta["done"]

    t0 = time.time()
    for k, item in enumerate(items):
        if item.id in done:
            continue
        ts = time.time()
        if any(b not in item.assets for b in BANDS):   # SR-only product: no ST
            done[item.id] = 0
            mm["ndvi"][k] = mm["ndbi"][k] = mm["lst"][k] = NODATA
            mm["water"][k] = -1
            print(f"  [{len(done):3d}/{n}] {item.id}  SKIPPED - missing "
                  f"{[b for b in BANDS if b not in item.assets]}", flush=True)
            continue
        nd, nb, ls, wa, nclear = scene_values(item, tr, w, h, union)
        mm["ndvi"][k], mm["ndbi"][k], mm["lst"][k], mm["water"][k] = nd, nb, ls, wa
        for a in mm.values():
            a.flush()
        done[item.id] = nclear
        json.dump(meta, open(meta_path, "w"))
        print(f"  [{len(done):3d}/{n}] {item.id}  clear {nclear / p:6.1%}  "
              f"{time.time() - ts:5.1f}s  elapsed {(time.time() - t0) / 60:5.1f} min",
              flush=True)

    # ------------------------------------------------ per-pixel median, chunked
    med = {k: np.full(p, np.nan, np.float32) for k in ("ndvi", "ndbi", "lst")}
    water_share = np.full(p, np.nan, np.float32)
    clear_obs = np.zeros(p, np.int32)
    for s in range(0, p, 20000):
        e = min(s + 20000, p)
        for k, scale in (("ndvi", 1e4), ("ndbi", 1e4), ("lst", 1e2)):
            blk = np.asarray(mm[k][:, s:e]).astype(np.float32)
            blk[blk == NODATA] = np.nan
            with np.errstate(all="ignore"):
                med[k][s:e] = np.nanmedian(blk, axis=0) / scale
        wb = np.asarray(mm["water"][:, s:e]).astype(np.float32)
        wb[wb < 0] = np.nan
        with np.errstate(all="ignore"):
            water_share[s:e] = np.nanmedian(wb, axis=0)
        clear_obs[s:e] = (np.asarray(mm["ndvi"][:, s:e]) != NODATA).sum(axis=0)

    pos = {v: i for i, v in enumerate(union)}
    rows = []
    for a, idx in zip(names, per):
        j = np.fromiter((pos[v] for v in idx), np.int64, len(idx))
        with np.errstate(all="ignore"):
            rows.append({
                "Address": a,
                "lst_day_c": round(float(np.nanmean(med["lst"][j])), 4),
                "ndvi": round(float(np.nanmean(med["ndvi"][j])), 5),
                "ndbi": round(float(np.nanmean(med["ndbi"][j])), 5),
                # GEE's median of a 0/1 band over an even count can be 0.5
                "water_share": round(float(np.nanmean(water_share[j])), 5),
                "clear_obs": round(float(clear_obs[j].mean()), 2),
                "pixels": int(len(j)),
            })
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    co = np.array([r["clear_obs"] for r in rows])
    print(f"\nscenes used {n}; clear obs per pixel, division means: "
          f"min {co.min():.1f} median {np.median(co):.1f} max {co.max():.1f}")
    print(f"nulls: " + ", ".join(f"{k}={sum(math.isnan(r[k]) for r in rows)}"
                                  for k in ("lst_day_c", "ndvi", "ndbi")))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())

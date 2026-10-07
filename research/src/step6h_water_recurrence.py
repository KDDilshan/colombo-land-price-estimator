"""Step 6h - ATTEMPT A at a working flood-frequency variable.

WHY THE FIRST ATTEMPT FAILED (flood_frequency_rejection.md)
  Global Flood Database, MODIS, 250 m: 159 of 194 divisions exactly 0, every
  non-zero value coastal, not one Kelani division above 0. The pixels were too
  coarse for an urban river corridor and the sea inside the 1 km buffers was
  scored as inundation.

THIS ATTEMPT
  JRC Global Surface Water 2021, `recurrence` band, 30 m - 8x finer, and it
  measures how often water RETURNS year to year, which is much closer to flood
  frequency than a count of large mapped events.

  THE MASK IS THE WHOLE POINT. Sea, rivers and lakes are water every year, so
  unmasked recurrence would rank the shoreline top and repeat the MODIS failure
  exactly. Pixels whose `occurrence` exceeds PERMANENT_PCT are permanent water
  and are removed from the buffer before averaging - removed from the
  denominator, not counted as zero, so a half-sea buffer is measured over its
  land portion rather than diluted by it.

  PERMANENT_PCT = 80 is set once, from the brief, before seeing any result.
  It is not tuned afterwards; tuning a mask until the geography looks right is
  fitting the instrument to the expectation.

VALIDATION IS GEOGRAPHIC, NOT STATISTICAL. The rejected MODIS column passed
both screening rules. The test here is whether the Kelani floodplain separates
from the dry southern uplands, and whether the coast stays out of the top.

Output: data/processed/water_recurrence_194.csv
        data/processed/water_recurrence_validation.md
"""

import csv
import datetime as dt
import math
import os

import numpy as np
import rasterio
from rasterio.windows import from_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
CENTROIDS = os.path.join(PROC, "gn_centroids.csv")
HAND = os.path.join(PROC, "gee_hand_floodfreq.csv")
FINAL = os.path.join(PROC, "final_dataset.csv")
OUT_CSV = os.path.join(PROC, "water_recurrence_194.csv")
OUT_MD = os.path.join(PROC, "water_recurrence_validation.md")

BUFFER_KM = 1.0
DEG_PER_KM_LAT = 1.0 / 110.574
PERMANENT_PCT = 80          # occurrence above this = sea, river, lake
GSW_NODATA = 255

BASE = ("https://storage.googleapis.com/global-surface-water/downloads2021/"
        "{band}/{band}_{lon:d}E_10Nv1_4_2021.tif")

# The geography test, fixed before the data was read.
EXPECT_HIGH = ["kolonnawa", "wellampitiya", "mulleriyawa", "kotikawatta",
               "sedawatta", "kelanimulla", "ambathale", "kaduwela"]
EXPECT_LOW = ["maharagama", "kottawa", "pannipitiya", "homagama", "nugegoda"]
COASTAL = ["fort", "uyana", "idama", "moratuwella", "mount lavinia",
           "koralawella", "kahapola"]

_open = {}


def src(url):
    if url not in _open:
        _open[url] = rasterio.open(url)
    return _open[url]


def deg_box(lat, lon, km):
    dlat = km * DEG_PER_KM_LAT
    dlon = km * DEG_PER_KM_LAT / max(math.cos(math.radians(lat)), 1e-6)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def circle_mask(t, shape, lat, lon, km):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    xs = t.c + (cols + 0.5) * t.a
    ys = t.f + (rows + 0.5) * t.e
    dy = (ys - lat) / DEG_PER_KM_LAT
    dx = (xs - lon) * math.cos(math.radians(lat)) / DEG_PER_KM_LAT
    return (dx * dx + dy * dy) <= km * km


def read_band(band, tlon, lat, lon):
    s = src(BASE.format(band=band, lon=tlon))
    w = from_bounds(*deg_box(lat, lon, BUFFER_KM), s.transform)
    arr = s.read(1, window=w, boundless=True, fill_value=GSW_NODATA)
    if arr.size == 0:
        return None, None
    return arr, circle_mask(s.window_transform(w), arr.shape, lat, lon, BUFFER_KM)


def tiles(lat, lon):
    x0, _, x1, _ = deg_box(lat, lon, BUFFER_KM)
    return sorted({int(math.floor(x / 10.0) * 10) for x in (x0, x1)})


def main():
    with open(CENTROIDS, encoding="utf-8-sig") as fh:
        pts = list(csv.DictReader(fh))
    with open(HAND, encoding="utf-8-sig") as fh:
        hand = {r["Address"]: float(r["hand_m"]) for r in csv.DictReader(fh)}
    with open(FINAL, encoding="utf-8-sig") as fh:
        fort = {}
        for r in csv.DictReader(fh):
            fort.setdefault(r["Address"], float(r["Distance from fort"]))

    rows = []
    for i, p in enumerate(pts, 1):
        lat, lon = float(p["gn_lat"]), float(p["gn_lon"])
        rec_vals, n_perm, n_total = [], 0, 0
        for tlon in tiles(lat, lon):
            try:
                rec, m = read_band("recurrence", tlon, lat, lon)
                occ, _ = read_band("occurrence", tlon, lat, lon)
            except Exception as exc:                      # noqa: BLE001
                print(f"  ! {p['Address']} tile {tlon}E: {type(exc).__name__}")
                continue
            if rec is None or occ is None:
                continue
            valid = m & (rec != GSW_NODATA) & (occ != GSW_NODATA)
            permanent = valid & (occ > PERMANENT_PCT)
            keep = valid & ~permanent
            n_total += int(valid.sum())
            n_perm += int(permanent.sum())
            rec_vals.extend(rec[keep].astype(float).tolist())

        rows.append({
            "Address": p["Address"],
            "Address_ID": p["Address_ID"],
            "water_recurrence_pct": (round(float(np.mean(rec_vals)), 4)
                                     if rec_vals else ""),
            "pixels_used": len(rec_vals),
            "pixels_permanent_masked": n_perm,
            "pct_buffer_permanent_water": (round(100.0 * n_perm / n_total, 2)
                                           if n_total else ""),
            "hand_m": round(hand.get(p["Address"], float("nan")), 3),
        })
        if i % 40 == 0 or i == len(pts):
            print(f"  [{i:3d}/{len(pts)}]")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report(rows, fort, hand)


def report(rows, fort, hand):
    ok = [r for r in rows if r["water_recurrence_pct"] != ""]
    v = [float(r["water_recurrence_pct"]) for r in ok]
    nulls = len(rows) - len(ok)
    addrs = [r["Address"] for r in ok]

    def pear(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    r_fort = pear(v, [fort[a] for a in addrs])
    r_hand = pear(v, [hand[a] for a in addrs])
    order = sorted(ok, key=lambda r: -float(r["water_recurrence_pct"]))
    rank = {r["Address"]: i + 1 for i, r in enumerate(order)}
    n = len(order)

    def group(names):
        got = [(x, rank[x], float(next(r["water_recurrence_pct"]
                                       for r in ok if r["Address"] == x)))
               for x in names if x in rank]
        return got, (float(np.mean([g[1] for g in got])) if got else float("nan"))

    hi, hi_mean = group(EXPECT_HIGH)
    lo, lo_mean = group(EXPECT_LOW)
    co, co_mean = group(COASTAL)

    separates = hi_mean < lo_mean and hi_mean < co_mean
    verdict = ("**PASS** - the Kelani floodplain ranks above both the dry uplands "
               "and the coast."
               if separates else
               "**FAIL** - the Kelani floodplain does not separate from the "
               "comparison groups.")

    print(f"\n{'='*70}\nATTEMPT A - JRC water recurrence, permanent water masked "
          f"at >{PERMANENT_PCT}% occurrence\n{'='*70}")
    print(f"n={len(ok)}  nulls={nulls}  distinct={len(set(v))}")
    print(f"min/mean/max = {min(v):.3f} / {np.mean(v):.3f} / {max(v):.3f}")
    print(f"r with Distance from fort = {r_fort:+.3f}")
    print(f"r with hand_m             = {r_hand:+.3f}")
    print(f"\nmean rank (1 = wettest of {n}):")
    print(f"  expected HIGH (Kelani)   {hi_mean:6.1f}")
    print(f"  expected LOW  (uplands)  {lo_mean:6.1f}")
    print(f"  coastal (must not top)   {co_mean:6.1f}")
    print(f"\n{verdict}\n")
    print(f"{'TOP 15':<26s}{'recur%':>9s}{'hand_m':>9s}   "
          f"{'BOTTOM 15':<26s}{'recur%':>9s}{'hand_m':>9s}")
    for a, b in zip(order[:15], order[-15:]):
        print(f"{a['Address']:<26s}{float(a['water_recurrence_pct']):9.3f}"
              f"{a['hand_m']:9.2f}   "
              f"{b['Address']:<26s}{float(b['water_recurrence_pct']):9.3f}"
              f"{b['hand_m']:9.2f}")

    md = ["# ATTEMPT A - `water_recurrence_pct` (JRC Global Surface Water, 30 m)",
          "", f"Run {dt.date.today().isoformat()} by "
          "`src/step6h_water_recurrence.py`.", "",
          "## Method", "",
          "- `recurrence` band, JRC Global Surface Water 2021, 30 m: how often "
          "water returns from year to year.",
          "- Mean over a 1 km circular buffer around each of the 194 centroids - "
          "the same geometry as every other hazard variable.",
          f"- **Permanent water masked out**: pixels with `occurrence` > "
          f"{PERMANENT_PCT}% are sea, river or lake and are removed from the "
          "buffer before averaging - removed from the denominator, not counted "
          "as zero.",
          f"- The {PERMANENT_PCT}% threshold was fixed before any result was "
          "seen and was not adjusted afterwards.", "",
          "## Statistics", "", "| statistic | value |", "|---|---|",
          f"| divisions | {len(ok)} |", f"| nulls | {nulls} |",
          f"| distinct values | {len(set(v))} |",
          f"| min / mean / max | {min(v):.3f} / {np.mean(v):.3f} / {max(v):.3f} |",
          f"| r vs `Distance from fort` | {r_fort:+.3f} |",
          f"| r vs `hand_m` | {r_hand:+.3f} |", "",
          "## Geography test - the one that decides it", "",
          "The groups below were fixed in the brief before the data was read. "
          "The rejected MODIS column passed both statistical screening rules and "
          "was still wrong, so the statistics above are not the test.", "",
          f"| group | mean rank (1 = wettest of {n}) |", "|---|---|",
          f"| expected HIGH - Kelani floodplain | **{hi_mean:.1f}** |",
          f"| expected LOW - southern uplands | {lo_mean:.1f} |",
          f"| coastal - must not dominate | {co_mean:.1f} |", "",
          verdict, "",
          "| expected HIGH | rank | value | | expected LOW | rank | value |",
          "|---|---|---|---|---|---|---|"]
    for i in range(max(len(hi), len(lo))):
        a = (f"{hi[i][0]} | {hi[i][1]} | {hi[i][2]:.2f}" if i < len(hi) else " | | ")
        b = (f"{lo[i][0]} | {lo[i][1]} | {lo[i][2]:.2f}" if i < len(lo) else " | | ")
        md.append(f"| {a} | | {b} |")
    md += ["", "| coastal check | rank | value |", "|---|---|---|"]
    for a, rk, val in co:
        md.append(f"| {a} | {rk} | {val:.2f} |")

    md += ["", "## Ranked extremes, with `hand_m` alongside", "",
           "| # | division | recurrence % | hand_m | | # | division | recurrence % | hand_m |",
           "|---|---|---|---|---|---|---|---|---|"]
    for i, (a, b) in enumerate(zip(order[:15], order[-15:]), 1):
        md.append(f"| {i} | {a['Address']} | {float(a['water_recurrence_pct']):.3f} "
                  f"| {a['hand_m']:.2f} | | {n - 15 + i} | {b['Address']} | "
                  f"{float(b['water_recurrence_pct']):.3f} | {b['hand_m']:.2f} |")
    md.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()

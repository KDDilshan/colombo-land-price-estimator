"""Step 28 - past flooding by GN division, from Survey Department of Sri Lanka
flood maps.

SOURCES (all Survey Department of Sri Lanka - government, official)
  2016  "Flooding Area Around Kelani River Basin (Part of) - May 2016",
        Flood_Map_2016_June_03_kelaniya.pdf. Field data 23 May - 2 June 2016
        by the Hydrographic Survey Unit, Sri Lanka Navy. The PDF is a GeoPDF:
        its viewport carries /GPTS lat-lon corners, and the flood extent is its
        own optional-content layer ("Flood_Area"). That layer alone is rendered,
        georeferenced from the corners and vectorised - no tracing by hand.
        Coverage = the map frame (79.85-80.18 E, 6.86-7.04 N); a GN outside it
        has NO RECORD for 2016, which is not the same as "not flooded".
  2018  "Flooding Area Around Kelani River Basin (Part of) - May 2018",
        Kelaniya_Flood_Map_2018-06-05.pdf. Field data 23 May - 1 June 2018 by
        the Special Surveys & Quality Control Branch, Survey Department. Same
        GeoPDF structure; the extent layer is named "Flood Boundary". Coverage =
        its map frame (79.84-80.14 E, 6.85-7.03 N).
  2025  "Kelani River Basin Flood Impact Analysis - Cyclone Ditwah, November
        2025" (completed January 2026), survey.gov.lk/sdweb/FooldMap2025/kelani.
        Flood extent from Sentinel-2 (30 Dec 2025), LiDAR and field verification,
        12,318.83 ha, published as a vector layer (FloodArea_2.js). Coverage =
        the Kelani basin boundary on the same site (RiverBasin_2.js).

UNIT.  The 443 GN polygons the app already uses
(model test/deploy/gn_boundaries.geojson; 187 modelled). Overlap is computed
geometrically, so no GN-name matching is involved. The Survey Department's own
2025 list of affected GNDs (Kelani_Admin_Boundaries_2025.xlsx) is used only as
an independent check of the overlap.

Output
  data/processed/official_flood_gn.csv          one row per GN polygon
  data/processed/official_flood_2016.geojson    the vectorised 2016 extent
  data/processed/official_flood_2018.geojson    the vectorised 2018 extent
  data/processed/official_flood_check.md        the 2025 list cross-check
"""

import json
import math
import os
import re

import numpy as np
import pandas as pd
import pymupdf
import rasterio.features
from affine import Affine
from shapely import STRtree, make_valid
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
GN = os.path.join(ROOT, "model test", "deploy", "gn_boundaries.geojson")
PDF16 = os.path.join(RAW, "survey_dept_2016", "Flood_Map_2016_June_03_kelaniya.pdf")
PDF18 = os.path.join(RAW, "survey_dept_2018", "Kelaniya_Flood_Map_2018-06-05.pdf")
GEOPDF = {  # year: (file, flood-layer name prefix, source line)
    "2016": (PDF16, "Flood_Area", "Survey Department of Sri Lanka, Flood Map 2016 "
             "(Kelaniya), field data 23 May-2 June 2016"),
    "2018": (PDF18, "Flood Boundary", "Survey Department of Sri Lanka, Flood Map 2018 "
             "(Kelaniya), field data 23 May-1 June 2018"),
}
F25 = os.path.join(RAW, "survey_dept_2025", "FloodArea_2.js")
B25 = os.path.join(RAW, "survey_dept_2025", "RiverBasin_2.js")
L25 = os.path.join(RAW, "survey_dept_2025", "Kelani_Admin_Boundaries_2025.xlsx")
DPI = 300
LAT0 = 6.9


def metres(g):
    """Local equirectangular metres - adequate for area shares inside Colombo."""
    kx = 111320.0 * math.cos(math.radians(LAT0))
    return transform(lambda x, y, z=None: (np.asarray(x) * kx, np.asarray(y) * 110574.0), g)


def read_js_geojson(path):
    t = open(path, encoding="utf-8").read()
    return json.loads(t[t.index("{"):].rstrip().rstrip(";"))


def flood_geopdf(pdf_path, layer_prefix):
    """Render only the flood layer of a Survey Department GeoPDF, georeference it
    from the /GPTS corners and vectorise it."""
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    raw = open(pdf_path, "rb").read()
    gpts = [float(v) for v in re.search(rb"/GPTS\[([^\]]+)\]", raw).group(1).split()]
    bbox = [float(v) for v in re.search(rb"/Viewport/BBox\[([^\]]+)\]", raw).group(1).split()]
    corners = list(zip(gpts[0::2], gpts[1::2]))          # (lat, lon) for LPTS
    lpts = [(0, 1), (0, 0), (1, 0), (1, 1)]              # (u, v); v = 0 at top

    flood_ui = [c["number"] for c in doc.layer_ui_configs() if c["text"].startswith(layer_prefix)]
    assert len(flood_ui) == 1, f"flood layer '{layer_prefix}' not found"
    keep = set(flood_ui) | {c["number"] for c in doc.layer_ui_configs()
                            if c["text"].startswith("Layers")}
    for c in doc.layer_ui_configs():
        if c["number"] not in keep:
            doc.set_layer_ui_config(c["number"], action=2)
    pix = page.get_pixmap(dpi=DPI)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    ink = (a.astype(int).sum(axis=2) < 3 * 250)          # anything not white
    s = DPI / 72.0
    H = page.rect.height
    x0, x1 = bbox[0] * s, bbox[2] * s
    ytop, ybot = (H - bbox[1]) * s, (H - bbox[3]) * s

    # least-squares affine: pixel (col,row) -> (lon, lat), from the 4 corners
    P, Q = [], []
    for (u, v), (lat, lon) in zip(lpts, corners):
        P.append([x0 + u * (x1 - x0), ytop + v * (ybot - ytop), 1.0])
        Q.append([lon, lat])
    coef, *_ = np.linalg.lstsq(np.array(P), np.array(Q), rcond=None)
    tr = Affine(coef[0, 0], coef[1, 0], coef[2, 0], coef[0, 1], coef[1, 1], coef[2, 1])
    resid = np.abs(np.array(P) @ coef - np.array(Q)).max()

    frame = np.zeros_like(ink)
    frame[int(ytop):int(ybot), int(x0):int(x1)] = True
    mask = (ink & frame).astype(np.uint8)
    polys = [shape(g) for g, v in rasterio.features.shapes(mask, mask=mask.astype(bool),
                                                          transform=tr) if v == 1]
    flood = make_valid(unary_union(polys)).buffer(0)
    cover = box(min(q[0] for q in Q), min(q[1] for q in Q),
                max(q[0] for q in Q), max(q[1] for q in Q))
    return flood, cover, dict(corner_fit_resid_deg=float(resid), pixels=int(mask.sum()),
                              area_ha=metres(flood).area / 1e4)


def main():
    gn = json.load(open(GN, encoding="utf-8"))
    g_geom = [make_valid(shape(f["geometry"])) for f in gn["features"]]
    g_prop = [f["properties"] for f in gn["features"]]

    geo, infos = {}, {}
    for yr, (path, layer, src) in GEOPDF.items():
        fl, cv, info = flood_geopdf(path, layer)
        geo[yr], infos[yr] = (fl, cv), info
        print(f"{yr}: {info}")
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"source": src},
             "geometry": mapping(fl.simplify(0.00005))}]},
            open(os.path.join(PROC, f"official_flood_{yr}.geojson"), "w"))

    f25 = make_valid(unary_union([shape(f["geometry"]) for f in read_js_geojson(F25)["features"]]))
    c25 = make_valid(unary_union([shape(f["geometry"]) for f in read_js_geojson(B25)["features"]]))
    print(f"2025: flood area {metres(f25).area / 1e4:,.1f} ha (published 12,318.83 ha)")

    rows = []
    for geom, p in zip(g_geom, g_prop):
        gm = metres(geom)
        area = gm.area
        rec = dict(address=p["address"], modelled=bool(p["modelled"]),
                   ds_divisions="; ".join(p.get("ds_divisions") or []),
                   adm4_members="; ".join(p.get("adm4_members") or []),
                   area_km2=round(area / 1e6, 4))
        for yr, (fl, cv) in list(geo.items()) + [("2025", (f25, c25))]:
            cov = metres(geom.intersection(cv)).area / area
            fr = metres(geom.intersection(fl)).area / area if cov > 0 else 0.0
            rec[f"covered_{yr}"] = round(cov, 4)
            rec[f"flooded_share_{yr}"] = round(fr, 4)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROC, "official_flood_gn.csv"), index=False)

    # ------------------------- cross-check against the Survey Department's list
    lst = pd.read_excel(L25, sheet_name="GND_by_DSD").ffill()
    lst = lst[lst.District.str.upper().str.strip() == "COLOMBO"]
    norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())
    listed = {norm(n) for n in lst.GND}
    df["listed_2025"] = [any(norm(m) in listed for m in a.split("; ") if m)
                         for a in df.adm4_members]
    matched = sum(df.listed_2025)
    col = df[df.covered_2025 > 0]
    tp = ((col.flooded_share_2025 > 0) & col.listed_2025).sum()
    fn = ((col.flooded_share_2025 == 0) & col.listed_2025).sum()
    fp = ((col.flooded_share_2025 > 0.01) & ~col.listed_2025).sum()
    md = ["# Official flood records by GN - cross-check", "",
          *[f"{yr} extent vectorised from the GeoPDF: {i['area_ha']:,.0f} ha inside the "
            f"map frame; corner-fit residual {i['corner_fit_resid_deg']:.2e} deg." for yr, i in infos.items()], "",
          f"2025 extent: {metres(f25).area / 1e4:,.1f} ha (Survey Department states "
          "12,318.83 ha).", "",
          f"Survey Department list of affected Colombo GNDs: {len(lst)} names; "
          f"{matched} of our {len(df)} GN polygons carry a member name on that list.", "",
          "Among GN polygons inside the 2025 basin coverage:", "",
          f"- on the list and overlapping the extent: **{tp}**",
          f"- on the list but no overlap: **{fn}**",
          f"- not on the list but > 1% of area flooded: **{fp}**", ""]
    open(os.path.join(PROC, "official_flood_check.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    for yr in ("2016", "2018", "2025"):
        c = df[df[f"covered_{yr}"] > 0]
        print(f"{yr}: {len(c)} GN polygons covered; flooded share > 0: "
              f"{(c[f'flooded_share_{yr}'] > 0).sum()}, > 10%: {(c[f'flooded_share_{yr}'] > 0.10).sum()}")


if __name__ == "__main__":
    main()

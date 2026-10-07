"""Loading and prediction for the Colombo land price model.

Everything is loaded once, at import, into module-level singletons. Nothing here
retrains or refits - the tuned XGBoost in deploy/model.pkl is used exactly as it
was saved, and every lookup table is read as given.

The model predicts log(price per perch); callers get the exponentiated value.

Importable and testable:

    import predictor
    predictor.predict(division="nugegoda", perches=10, land_type="Land_type_Residential")
    predictor.locate(6.8771, 79.8903)
"""

import json
import math
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(HERE, "deploy")

PERCH_M2 = 25.29285          # 1 perch in square metres
EARTH_RADIUS_M = 6371008.8   # IUGG mean radius

# Performance figures quoted to the user, for the model in deploy/: the tuned
# XGBoost (the thesis main model) on 50 features over 6,167 listings in 187
# divisions, scored on the random 80/20 held-out split (seed 42, 1,234 listings).
# Written by src/step33_deploy_xgb.py to data/processed/deploy_xgb_metrics.json;
# re-run it after every retrain. The tuned Random Forest served before 2026-09-28
# (R2 89.46%, median error 6.4%) is in deploy/_backup_rf_20260928.
MEDIAN_APE = 7.2          # on the rupee scale after np.exp
MEAN_APE = 15.4           # on the rupee scale after np.exp
# Only 31 of the 7,034 training listings are under 6 perches (smallest is 3), so the
# model prices plots below this size from very little data - flagged, not blocked.
SMALL_PLOT_PERCHES = 6
SMALL_PLOT_LISTINGS = 31
# One price per perch per division and land type: the model is always asked for a
# standard 12-perch plot, so the per-perch price does not change with the size the
# user enters - only the total does (total = price per perch x perches).
REFERENCE_PERCHES = 12.0
R2_KNOWN = 90.11


# --------------------------------------------------------------------- loading

def _load():
    model = joblib.load(os.path.join(DEPLOY, "model.pkl"))

    with open(os.path.join(DEPLOY, "feature_order.json"), encoding="utf-8") as fh:
        feature_order = json.load(fh)

    with open(os.path.join(DEPLOY, "fallback.json"), encoding="utf-8") as fh:
        fallback = json.load(fh)

    enc = pd.read_csv(os.path.join(DEPLOY, "address_encoding.csv"))
    enc["Address"] = enc["Address"].str.strip().str.lower()
    encoding = dict(zip(enc["Address"], enc["Address_target_enc"]))

    values = pd.read_csv(os.path.join(DEPLOY, "division_values.csv"), index_col=0)
    values.index = values.index.str.strip().str.lower()

    # The Colombo-wide row for divisions the model never saw
    global_row = values.median()

    with open(os.path.join(DEPLOY, "gn_boundaries.geojson"), encoding="utf-8") as fh:
        boundaries = json.load(fh)

    return model, feature_order, fallback, encoding, values, global_row, boundaries


MODEL, FEATURE_ORDER, FALLBACK, ENCODING, DIVISION_VALUES, GLOBAL_ROW, BOUNDARIES = _load()

GLOBAL_MEAN_LOG_PRICE = FALLBACK["global_mean_log_price"]

# Land types are read from the model's own feature list, never hardcoded
LAND_TYPE_COLUMNS = [c for c in FEATURE_ORDER if c.startswith("Land_type_")]

DEFAULT_LAND_TYPE = ("Land_type_Residential" if "Land_type_Residential" in LAND_TYPE_COLUMNS
                     else LAND_TYPE_COLUMNS[0])

FLOOD_FIELDS = ("water_occurrence_pct", "hand_m", "dist_to_stream_km", "ndvi")


def _load_flood_records():
    """Past flooding per GN polygon, from official records only (src/step30):
    Survey Department flood maps (May 2016; May 2018; Cyclone Ditwah, Nov 2025) and the
    DMC DesInventar database (1974-2019). Descriptive - never a model input."""
    path = os.path.join(DEPLOY, "flood_records.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    return {r["address"].strip().lower(): r for r in df.to_dict("records")}


FLOOD_RECORDS = _load_flood_records()

FLOOD_SOURCES = [
    "Survey Department of Sri Lanka, Flood Map 2016 - Kelaniya area "
    "(field data 23 May-2 June 2016)",
    "Survey Department of Sri Lanka, Flood Map 2018 - Kelaniya area "
    "(field data 23 May-1 June 2018)",
    "Survey Department of Sri Lanka, Kelani River Basin Flood Impact Analysis - "
    "Cyclone Ditwah, November 2025",
    "Disaster Management Centre, DesInventar disaster database (records 1974-2019)",
]


def flood_record(gn_address):
    """Official past-flooding record for one GN polygon, or None if unknown.

    A map share is None when the GN lies mostly outside that map's coverage -
    'not mapped' is different from 'mapped and not flooded'.
    """
    r = FLOOD_RECORDS.get(str(gn_address or "").strip().lower())
    if r is None:
        return None

    def share(year):
        return (round(float(r[f"flooded_share_{year}"]) * 100, 1)
                if float(r[f"covered_{year}"]) > 0.5 else None)

    years = [int(y) for y in str(r["dmc_years"]).split()]
    return {
        "indication": r["indication"],
        "flooded_pct_2016": share("2016"),
        "flooded_pct_2018": share("2018"),
        "flooded_pct_2025": share("2025"),
        "dmc_years": years,
        "dmc_ds_flood_records": int(r["dmc_ds_flood_records"]),
        "ds_division": r["ds_divisions"],
        "sources": FLOOD_SOURCES,
    }


def land_type_label(column):
    """'Land_type_Agricultural_Commercial' -> 'Agricultural / Commercial'."""
    return column[len("Land_type_"):].replace("_", " / ")


LAND_TYPES = [{"value": c, "label": land_type_label(c)} for c in LAND_TYPE_COLUMNS]

KNOWN_DIVISIONS = sorted(DIVISION_VALUES.index)
DIVISION_COUNT = len(KNOWN_DIVISIONS)


def confidence_grade(n_listings):
    """Data-support grade: how much evidence a division's row actually rests on.

    This is not a calibrated confidence interval and must not be read as one -
    the thresholds are round numbers keyed to listing count, not to error. The
    function and the JSON field keep the name 'confidence' so that existing
    clients continue to work; the thesis calls it a data-support grade.
    """
    if n_listings >= 50:
        return "high"
    if n_listings >= 10:
        return "medium"
    if n_listings >= 1:
        return "low"
    return "none"


# ------------------------------------------------------- point-in-polygon lookup

def _prepare_index():
    """Flatten the boundary file into (bbox, rings, properties) for fast testing."""
    modelled, other = [], []
    for feat in BOUNDARIES["features"]:
        props = feat["properties"]
        rings = [poly[0] for poly in feat["geometry"]["coordinates"]]
        entry = (props["bbox"], rings, props)
        (modelled if props["modelled"] else other).append(entry)
    return modelled, other


_MODELLED_INDEX, _OTHER_INDEX = _prepare_index()


def _point_in_ring(lon, lat, ring):
    """Standard ray-casting test. Points on the edge may fall either way."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _search(index, lat, lon):
    for bbox, rings, props in index:
        if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        for ring in rings:
            if _point_in_ring(lon, lat, ring):
                return props
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a)) / 1000.0


def _nearest_modelled(lat, lon):
    best, best_km = None, None
    for _bbox, _rings, props in _MODELLED_INDEX:
        km = _haversine_km(lat, lon, props["gn_lat"], props["gn_lon"])
        if best_km is None or km < best_km:
            best, best_km = props, km
    return best, best_km


def locate(lat, lon):
    """Turn a clicked point into a GN division.

    Returns a dict with:
      division   the modelled division name, or None if the point is not in one
      match      'polygon' | 'nearest' | 'outside'
      known      whether `division` is one of the DIVISION_COUNT modelled divisions
      adm4_name  the actual GN division the point falls in, when it can be named
    """
    hit = _search(_MODELLED_INDEX, lat, lon)
    if hit is not None:
        return {
            "division": hit["address"],
            "gn_address": hit["address"],
            "match": "polygon",
            "known": True,
            "adm4_name": ", ".join(hit["adm4_members"]),
            "ds_division": ", ".join(hit["ds_divisions"]),
            "distance_km": 0.0,
        }

    other = _search(_OTHER_INDEX, lat, lon)
    nearest, km = _nearest_modelled(lat, lon)
    if other is not None:
        # Inside Colombo district, but in a division the model can't currently price.
        # Two different reasons look the same on the map but aren't: a division that
        # was never in the training data (modelled_reason absent) vs. one that was
        # modelled before this retrain and got dropped (modelled_reason set, see
        # build_boundaries.py) - worth telling the user apart.
        return {
            "division": None,
            "gn_address": other["address"],
            "match": "outside",
            "known": False,
            "adm4_name": ", ".join(other["adm4_members"]),
            "ds_division": ", ".join(other["ds_divisions"]),
            "modelled_reason": other.get("modelled_reason"),
            "nearest_division": nearest["address"],
            "distance_km": round(km, 2),
        }

    return {
        "division": None,
        "match": "outside",
        "known": False,
        "adm4_name": None,
        "ds_division": None,
        "modelled_reason": None,
        "nearest_division": nearest["address"],
        "distance_km": round(km, 2),
    }


def boundary_of(division, tolerance=1e-4):
    """Simplified outline of a modelled division, for drawing on the map."""
    for _bbox, rings, props in _MODELLED_INDEX:
        if props["address"] == division:
            return {
                "type": "MultiPolygon",
                "coordinates": [[_simplify(r, tolerance)] for r in rings],
            }
    return None


def _simplify(ring, tolerance):
    """Douglas-Peucker, iterative. Display only - the exact ring is used for lookup."""
    if len(ring) < 4:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        start, end = stack.pop()
        x1, y1 = ring[start]
        x2, y2 = ring[end]
        dx, dy = x2 - x1, y2 - y1
        denom = math.hypot(dx, dy)
        far_i, far_d = None, tolerance
        for i in range(start + 1, end):
            x0, y0 = ring[i]
            if denom == 0:
                d = math.hypot(x0 - x1, y0 - y1)
            else:
                d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
            if d > far_d:
                far_i, far_d = i, d
        if far_i is not None:
            keep[far_i] = True
            stack.append((start, far_i))
            stack.append((far_i, end))
    out = [p for p, k in zip(ring, keep) if k]
    return out if len(out) >= 4 else ring


# ------------------------------------------------------------------ plot area

def polygon_area_m2(latlngs):
    """Area of a lat/lon ring on a sphere of radius EARTH_RADIUS_M.

    Not a naive lat/lon area - each edge contributes its true spherical wedge, so
    the result is correct at Colombo's latitude rather than inflated by cos(lat).
    This is a spherical-earth calculation, not an ellipsoidal geodesic one; at
    parcel scale the difference is far below the precision the interface needs.
    """
    pts = list(latlngs)
    if len(pts) < 3:
        return 0.0
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(pts, pts[1:]):
        l1, l2 = math.radians(lon1), math.radians(lon2)
        p1, p2 = math.radians(lat1), math.radians(lat2)
        total += (l2 - l1) * (2 + math.sin(p1) + math.sin(p2))
    return abs(total * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0)


def m2_to_perches(area_m2):
    return area_m2 / PERCH_M2


# ------------------------------------------------------------------ prediction

def _flood_profile(row):
    return {f: (None if pd.isna(row.get(f)) else float(row[f])) for f in FLOOD_FIELDS}


def predict(division=None, perches=None, land_type=None, lat=None, lon=None):
    """Predict price per perch for a plot.

    `division` may be given directly, or `lat`/`lon` looked up against the GN
    boundaries. A division outside DIVISION_COUNT falls back to Colombo-wide medians
    and the global mean target encoding, and is reported as known=False.
    """
    if perches is None or float(perches) <= 0:
        raise ValueError("perches must be a positive number")
    perches = float(perches)

    land_type = land_type or DEFAULT_LAND_TYPE
    if land_type not in LAND_TYPE_COLUMNS:
        raise ValueError(f"unknown land type: {land_type}")

    located = None
    if division is None and lat is not None and lon is not None:
        located = locate(float(lat), float(lon))
        division = located["division"] or located.get("nearest_division")

    if division is None:
        raise ValueError("a division, or a lat/lon inside Colombo, is required")

    division = str(division).strip().lower()
    known = division in DIVISION_VALUES.index
    if located is not None and located["match"] == "outside":
        known = False

    # ---- assemble the feature row exactly as the training pipeline ordered it
    if known:
        row = DIVISION_VALUES.loc[division].copy()
        row["Address_target_enc"] = ENCODING[division]
        n_listings = int(row["listing_count"])
    else:
        row = GLOBAL_ROW.copy()
        row["Address_target_enc"] = GLOBAL_MEAN_LOG_PRICE
        n_listings = 0

    row["Land_size_Perches_"] = REFERENCE_PERCHES
    for col in LAND_TYPE_COLUMNS:
        row[col] = 0.0
    row[land_type] = 1.0

    X = pd.DataFrame([row]).reindex(columns=FEATURE_ORDER, fill_value=0)
    log_price = float(MODEL.predict(X)[0])
    price_per_perch = float(np.exp(log_price))

    result = {
        "division": division,
        "known_division": bool(known),
        "listing_count": n_listings,
        "confidence": confidence_grade(n_listings),
        "perches": perches,
        "small_plot": perches < SMALL_PLOT_PERCHES,
        "land_type": land_type,
        "land_type_label": land_type_label(land_type),
        "price_per_perch": price_per_perch,
        "total_price": price_per_perch * perches,
        "log_price": log_price,
        "flood": _flood_profile(row),
        # the GN actually clicked, even when its price falls back to a neighbour
        "flood_record": flood_record(located.get("gn_address") if located is not None
                                     else division),
        "accuracy": {
            "median_ape": MEDIAN_APE,
            "mean_ape": MEAN_APE,
            "r2_known": R2_KNOWN,
        },
    }
    if located is not None:
        result["located"] = located
    return result

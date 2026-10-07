"""Nearby amenities for a plot, via the OpenStreetMap Overpass API.

Results are cached in AmenityCache keyed by coordinate rounded to ~100 m, both to
stay well inside Overpass's fair-use limits and to keep repeat lookups fast. This
is the paid-plan differentiator: predictor.py never sees or uses this data, it
only supplements the price estimate for the user.
"""

import math
from datetime import datetime, timedelta

import requests

from .extensions import db
from .models import AmenityCache

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 1500
CACHE_TTL_DAYS = 30
BUCKET = 0.001  # ~111 m at the equator - fine-grained enough for "nearby"
EARTH_RADIUS_M = 6371008.8

CATEGORIES = {
    "school": '["amenity"="school"]',
    "hospital": '["amenity"~"^(hospital|clinic|doctors)$"]',
    "supermarket": '["shop"~"^(supermarket|convenience)$"]',
    "bank": '["amenity"~"^(bank|atm)$"]',
    "bus_stop": '["highway"="bus_stop"]',
    "railway_station": '["railway"="station"]',
    "park": '["leisure"="park"]',
}

CATEGORY_LABELS = {
    "school": "Schools",
    "hospital": "Hospitals & clinics",
    "supermarket": "Supermarkets",
    "bank": "Banks & ATMs",
    "bus_stop": "Bus stops",
    "railway_station": "Railway stations",
    "park": "Parks",
}


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _bucket(lat, lon):
    return round(lat / BUCKET) * BUCKET, round(lon / BUCKET) * BUCKET


def _categorize(tags):
    amenity, shop, highway, railway, leisure = (
        tags.get(k) for k in ("amenity", "shop", "highway", "railway", "leisure")
    )
    if amenity == "school":
        return "school"
    if amenity in ("hospital", "clinic", "doctors"):
        return "hospital"
    if shop in ("supermarket", "convenience"):
        return "supermarket"
    if amenity in ("bank", "atm"):
        return "bank"
    if highway == "bus_stop":
        return "bus_stop"
    if railway == "station":
        return "railway_station"
    if leisure == "park":
        return "park"
    return None


def _query_overpass(lat, lon):
    clauses = "".join(f"node{sel}(around:{RADIUS_M},{lat},{lon});" for sel in CATEGORIES.values())
    query = f"[out:json][timeout:20];({clauses});out center;"
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=25,
        headers={"User-Agent": "colombo-land-price-estimator/1.0"},
    )
    resp.raise_for_status()
    return resp.json().get("elements", [])


def _fetch(lat, lon):
    elements = _query_overpass(lat, lon)
    grouped = {cat: [] for cat in CATEGORIES}
    for el in elements:
        cat = _categorize(el.get("tags", {}))
        if cat is None:
            continue
        elat, elon = el.get("lat"), el.get("lon")
        if elat is None or elon is None:
            continue
        grouped[cat].append(
            {
                "name": el.get("tags", {}).get("name") or CATEGORY_LABELS[cat].rstrip("s"),
                "lat": elat,
                "lon": elon,
                "distance_m": round(_haversine_m(lat, lon, elat, elon)),
            }
        )
    for cat, items in grouped.items():
        items.sort(key=lambda x: x["distance_m"])
        grouped[cat] = items[:8]
    return grouped


def nearby(lat, lon):
    """Grouped, distance-sorted amenities near (lat, lon). Cached per ~100 m bucket."""
    blat, blon = _bucket(lat, lon)
    cached = AmenityCache.query.filter_by(lat_bucket=blat, lon_bucket=blon).first()
    if cached and cached.fetched_at > datetime.utcnow() - timedelta(days=CACHE_TTL_DAYS):
        return cached.payload

    try:
        grouped = _fetch(lat, lon)
    except requests.RequestException:
        return cached.payload if cached else {cat: [] for cat in CATEGORIES}

    if cached:
        cached.payload = grouped
        cached.fetched_at = datetime.utcnow()
    else:
        db.session.add(AmenityCache(lat_bucket=blat, lon_bucket=blon, payload=grouped))
    db.session.commit()
    return grouped

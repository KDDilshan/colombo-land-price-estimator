"""JSON API for the estimator. Wraps predictor.py (untouched) with the SaaS
concerns: login, free-tier quota, saved history, and the paid amenities lookup.
"""

import traceback

from flask import Blueprint, current_app, jsonify, request, url_for
from flask_login import current_user, login_required

import predictor
from .. import amenities
from ..extensions import db
from ..models import EstimateHistory

api_bp = Blueprint("api", __name__)


@api_bp.post("/predict")
@login_required
def api_predict():
    limit = current_app.config["FREE_TIER_MONTHLY_LIMIT"]
    if not current_user.is_pro and current_user.estimates_this_period() >= limit:
        return jsonify({
            "error": "quota_exceeded",
            "message": f"Free plan is limited to {limit} estimates every 30 days. "
                       "Upgrade to Pro for unlimited estimates.",
            "upgrade_url": url_for("billing.pricing"),
        }), 402

    payload = request.get_json(silent=True) or {}

    try:
        perches = float(payload.get("perches"))
    except (TypeError, ValueError):
        return jsonify({"error": "perches must be a number"}), 400
    if not perches > 0:
        return jsonify({"error": "perches must be greater than zero"}), 400

    division = payload.get("division") or None
    lat, lon = payload.get("lat"), payload.get("lon")
    if division is None and (lat is None or lon is None):
        return jsonify({"error": "click a point on the map, or pick a division"}), 400

    try:
        result = predictor.predict(
            division=division,
            lat=lat,
            lon=lon,
            perches=perches,
            land_type=payload.get("land_type"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:                                    # pragma: no cover
        traceback.print_exc()
        return jsonify({"error": "prediction failed"}), 500

    if result["known_division"]:
        result["boundary"] = predictor.boundary_of(result["division"])

    # Anchor point for the amenities lookup: the clicked/drawn point if we have
    # one, else the centroid of the resolved division's boundary.
    anchor_lat, anchor_lon = lat, lon
    if anchor_lat is None and result.get("boundary"):
        ring = result["boundary"]["coordinates"][0][0]
        anchor_lon = sum(p[0] for p in ring) / len(ring)
        anchor_lat = sum(p[1] for p in ring) / len(ring)

    estimate = EstimateHistory(
        user_id=current_user.id,
        division=result["division"],
        lat=anchor_lat,
        lon=anchor_lon,
        perches=perches,
        land_type=result["land_type"],
        land_type_label=result["land_type_label"],
        price_per_perch=result["price_per_perch"],
        total_price=result["total_price"],
        known_division=result["known_division"],
        confidence=result["confidence"],
    )
    db.session.add(estimate)
    db.session.commit()

    result["estimate_id"] = estimate.id
    result["anchor"] = {"lat": anchor_lat, "lon": anchor_lon} if anchor_lat is not None else None
    result["is_pro"] = current_user.is_pro
    result["remaining_free"] = None if current_user.is_pro else max(0, limit - current_user.estimates_this_period())

    return jsonify(result)


@api_bp.post("/compare")
@login_required
def api_compare():
    """Price the same plot under every valid land-type category, so the price
    difference between e.g. Residential and Commercial is visible directly."""
    if not current_user.is_pro:
        return jsonify({
            "error": "upgrade_required",
            "message": "Comparing land types is a Pro feature.",
            "upgrade_url": url_for("billing.pricing"),
        }), 402

    payload = request.get_json(silent=True) or {}

    try:
        perches = float(payload.get("perches"))
    except (TypeError, ValueError):
        return jsonify({"error": "perches must be a number"}), 400
    if not perches > 0:
        return jsonify({"error": "perches must be greater than zero"}), 400

    division = payload.get("division") or None
    lat, lon = payload.get("lat"), payload.get("lon")
    if division is None and (lat is None or lon is None):
        return jsonify({"error": "click a point on the map, or pick a division"}), 400

    results = []
    known_division = None
    for land_type in predictor.LAND_TYPE_COLUMNS:
        try:
            # A division is only pinned for reuse once it's a genuine match
            # (known=True) - re-resolving the same lat/lon each time is cheap
            # and, for a point outside Colombo, is required for correctness:
            # predict() still labels an unknown/fallback result with the
            # *nearest* division as a location hint even though it wasn't
            # priced with that division's data, so pinning on it here would
            # silently swap every later land type from the Colombo-wide
            # fallback onto that nearby division's real (and very different)
            # listing values.
            r = predictor.predict(
                division=division, lat=lat if division is None else None,
                lon=lon if division is None else None,
                perches=perches, land_type=land_type,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if r["known_division"]:
            division = r["division"]
        known_division = r["known_division"]
        results.append({
            "land_type": land_type,
            "land_type_label": r["land_type_label"],
            "price_per_perch": r["price_per_perch"],
            "total_price": r["total_price"],
        })

    results.sort(key=lambda r: r["price_per_perch"])
    return jsonify({
        "division": division,
        "known_division": known_division,
        "perches": perches,
        "results": results,
    })


@api_bp.post("/amenities")
@login_required
def api_amenities():
    if not current_user.is_pro:
        return jsonify({
            "error": "upgrade_required",
            "message": "Nearby amenities are a Pro feature.",
            "upgrade_url": url_for("billing.pricing"),
        }), 402

    payload = request.get_json(silent=True) or {}
    lat, lon = payload.get("lat"), payload.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon must be numbers"}), 400

    try:
        grouped = amenities.nearby(lat, lon)
    except Exception:                                    # pragma: no cover
        traceback.print_exc()
        return jsonify({"error": "amenities lookup failed"}), 502

    return jsonify({
        "categories": {
            cat: {"label": amenities.CATEGORY_LABELS[cat], "items": items}
            for cat, items in grouped.items()
        }
    })

from collections import Counter, defaultdict
from datetime import datetime

from flask import Blueprint, abort, render_template, send_file
from flask_login import current_user, login_required

import predictor

from ..models import EstimateHistory
from .pdf import build_pdf

dashboard_bp = Blueprint("dashboard", __name__)


FLOOD_LEVELS = ["High", "Medium", "Low", "No official record"]


def _dashboard_stats(estimates):
    """Aggregates for the dashboard charts, from the user's own estimate history.

    Everything here summarises what the user has already priced - it adds no
    model output of its own. The flood breakdown reads the same official
    past-flooding record the estimate panel shows (predictor.flood_record).
    """
    if not estimates:
        return None
    prices = sorted(e.price_per_perch for e in estimates)
    n = len(prices)
    median = prices[n // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2

    by_div = defaultdict(list)
    for e in estimates:
        if e.known_division and e.division:
            by_div[e.division].append(e.price_per_perch)
    divisions = sorted(by_div.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:8]
    division_bars = sorted(
        ({"label": d.title(), "value": sum(v) / len(v), "count": len(v)} for d, v in divisions),
        key=lambda r: -r["value"])

    types = Counter((e.land_type_label or "Unknown") for e in estimates).most_common()
    land_types = [{"label": k, "value": v} for k, v in types[:4]]
    if len(types) > 4:
        land_types.append({"label": "Other", "value": sum(v for _, v in types[4:])})

    today = datetime.utcnow().date().replace(day=1)
    months = []
    y, m = today.year, today.month
    for _ in range(6):
        months.append((y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    months.reverse()
    per_month = Counter((e.created_at.year, e.created_at.month) for e in estimates)
    activity = [{"label": datetime(yy, mm, 1).strftime("%b %Y"), "value": per_month.get((yy, mm), 0)}
                for yy, mm in months]

    flood = Counter()
    for e in estimates:
        rec = predictor.flood_record(e.division) if e.division else None
        flood[rec["indication"] if rec else "No official record"] += 1
    flood_bars = [{"label": k, "value": flood.get(k, 0)} for k in FLOOD_LEVELS]

    return {
        "count": n,
        "median_price": median,
        "total_value": sum(e.total_price for e in estimates),
        "division_count": len({e.division for e in estimates if e.division}),
        "division_bars": division_bars,
        "land_types": land_types,
        "activity": activity,
        "flood": flood_bars,
    }


@dashboard_bp.get("/")
@login_required
def index():
    estimates = (
        current_user.estimates.order_by(EstimateHistory.created_at.desc()).limit(100).all()
    )
    return render_template("dashboard/index.html", estimates=estimates,
                           stats=_dashboard_stats(estimates))


@dashboard_bp.get("/account")
@login_required
def account():
    return render_template("dashboard/account.html")


@dashboard_bp.get("/estimate/<int:estimate_id>/pdf")
@login_required
def estimate_pdf(estimate_id):
    if not current_user.is_pro:
        abort(402)

    estimate = current_user.estimates.filter_by(id=estimate_id).first()
    if estimate is None:
        abort(404)

    buf = build_pdf(estimate)
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"estimate-{estimate.id}.pdf",
    )

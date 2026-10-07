from flask import Blueprint, abort, current_app, flash, redirect, render_template, url_for
from flask_login import current_user

from ..extensions import db
from ..models import EstimateHistory, User

admin_bp = Blueprint("admin", __name__)


@admin_bp.before_request
def require_admin():
    if not current_user.is_authenticated:
        return current_app.login_manager.unauthorized()
    if not current_user.is_admin:
        abort(403)


@admin_bp.get("/")
def index():
    users = User.query.order_by(User.created_at.desc()).all()

    # per-user estimate counts, in one query rather than one per row
    counts = {}
    for user_id, in EstimateHistory.query.with_entities(EstimateHistory.user_id).all():
        counts[user_id] = counts.get(user_id, 0) + 1

    return render_template("admin/index.html", users=users, counts=counts)


@admin_bp.post("/users/<int:user_id>/toggle-pro")
def toggle_pro(user_id):
    user = db.get_or_404(User, user_id)

    # A manual override for support/testing - it does NOT touch Stripe, so a
    # user with a real subscription keeps being billed even if revoked here,
    # and granting here doesn't create a Stripe subscription for them.
    if user.plan == "pro":
        user.plan = "free"
        user.subscription_status = "canceled"
    else:
        user.plan = "pro"
        user.subscription_status = "active"

    db.session.commit()
    flash(f"{user.email} is now {'Pro' if user.plan == 'pro' else 'Free'} (manual override).", "info")
    return redirect(url_for("admin.index"))

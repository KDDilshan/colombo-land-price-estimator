from datetime import datetime

import stripe
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from ..extensions import csrf, db
from ..models import User
from .stripe_client import client

billing_bp = Blueprint("billing", __name__)


@billing_bp.get("/pricing")
def pricing():
    return render_template(
        "pricing.html",
        free_limit=current_app.config["FREE_TIER_MONTHLY_LIMIT"],
        stripe_configured=bool(current_app.config.get("STRIPE_PRICE_PRO_MONTHLY")),
    )


@billing_bp.post("/checkout")
@login_required
def checkout():
    if not current_app.config.get("STRIPE_PRICE_PRO_MONTHLY"):
        return "Billing isn't configured yet - add Stripe keys to .env.", 503

    s = client()
    if not current_user.stripe_customer_id:
        customer = s.Customer.create(email=current_user.email, name=current_user.name)
        current_user.stripe_customer_id = customer.id
        db.session.commit()

    session = s.checkout.Session.create(
        customer=current_user.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": current_app.config["STRIPE_PRICE_PRO_MONTHLY"], "quantity": 1}],
        success_url=url_for("dashboard.account", _external=True) + "?upgraded=1",
        cancel_url=url_for("billing.pricing", _external=True),
    )
    return redirect(session.url, code=303)


@billing_bp.post("/portal")
@login_required
def portal():
    if not current_user.stripe_customer_id:
        return redirect(url_for("billing.pricing"))

    s = client()
    session = s.billing_portal.Session.create(
        customer=current_user.stripe_customer_id,
        return_url=url_for("dashboard.account", _external=True),
    )
    return redirect(session.url, code=303)


@csrf.exempt
@billing_bp.post("/webhook")
def webhook():
    # Stripe authenticates this request via its own signature (verified below),
    # not our session-based CSRF token, so it's exempted from CSRF checking.
    s = client()
    payload = request.data
    sig = request.headers.get("Stripe-Signature", "")

    try:
        event = s.Webhook.construct_event(payload, sig, current_app.config["STRIPE_WEBHOOK_SECRET"])
    except (ValueError, stripe.error.SignatureVerificationError):
        return "invalid signature", 400

    obj = event["data"]["object"]
    etype = event["type"]

    if etype == "checkout.session.completed":
        user = User.query.filter_by(stripe_customer_id=obj.get("customer")).first()
        if user:
            user.stripe_subscription_id = obj.get("subscription")
            db.session.commit()

    elif etype in ("customer.subscription.created", "customer.subscription.updated"):
        user = User.query.filter_by(stripe_customer_id=obj.get("customer")).first()
        if user:
            user.stripe_subscription_id = obj.get("id")
            user.subscription_status = obj.get("status")
            user.plan = "pro" if obj.get("status") in ("active", "trialing") else "free"
            period_end = obj.get("current_period_end")
            if period_end:
                user.current_period_end = datetime.utcfromtimestamp(period_end)
            db.session.commit()

    elif etype == "customer.subscription.deleted":
        user = User.query.filter_by(stripe_customer_id=obj.get("customer")).first()
        if user:
            user.plan = "free"
            user.subscription_status = "canceled"
            db.session.commit()

    return jsonify({"received": True})

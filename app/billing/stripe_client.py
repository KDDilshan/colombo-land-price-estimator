import stripe
from flask import current_app


def client():
    stripe.api_key = current_app.config["STRIPE_SECRET_KEY"]
    return stripe

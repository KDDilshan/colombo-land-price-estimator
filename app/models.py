from datetime import datetime, timedelta

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255))
    password_hash = db.Column(db.String(255), nullable=True)
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    stripe_customer_id = db.Column(db.String(255))
    stripe_subscription_id = db.Column(db.String(255))
    plan = db.Column(db.String(20), default="free", nullable=False)
    subscription_status = db.Column(db.String(30))
    current_period_end = db.Column(db.DateTime)

    estimates = db.relationship(
        "EstimateHistory", backref="user", lazy="dynamic", cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return bool(self.password_hash) and check_password_hash(self.password_hash, password)

    @property
    def is_pro(self):
        return self.plan == "pro" and self.subscription_status in ("active", "trialing")

    def estimates_this_period(self, days=30):
        cutoff = datetime.utcnow() - timedelta(days=days)
        return self.estimates.filter(EstimateHistory.created_at >= cutoff).count()


class EstimateHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    division = db.Column(db.String(255))
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    perches = db.Column(db.Float, nullable=False)
    land_type = db.Column(db.String(64))
    land_type_label = db.Column(db.String(64))

    price_per_perch = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    known_division = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.String(16))

    label = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


class AmenityCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lat_bucket = db.Column(db.Float, nullable=False)
    lon_bucket = db.Column(db.Float, nullable=False)
    payload = db.Column(db.JSON, nullable=False)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.UniqueConstraint("lat_bucket", "lon_bucket", name="uq_amenity_bucket"),)

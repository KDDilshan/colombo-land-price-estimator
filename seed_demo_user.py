"""Create (or reset) a local Pro demo account for thesis screenshots. Dev database only.

    venv/Scripts/python.exe seed_demo_user.py      writes the credentials to .demo_user
"""
import secrets

from app import create_app
from app.extensions import db
from app.models import User

EMAIL = "thesis.demo@example.test"

app = create_app()
with app.app_context():
    pw = secrets.token_urlsafe(12)
    u = User.query.filter_by(email=EMAIL).first() or User(email=EMAIL, name="Thesis Demo")
    u.set_password(pw)
    u.plan, u.subscription_status = "pro", "active"
    db.session.add(u)
    db.session.commit()
    with open(".demo_user", "w") as f:
        f.write(f"{EMAIL}\n{pw}\n")
    print("demo user ready:", EMAIL, "(password in .demo_user)")

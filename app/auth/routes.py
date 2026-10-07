from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..extensions import db, limiter, oauth
from ..models import User

auth_bp = Blueprint("auth", __name__)

RESET_SALT = "password-reset"
RESET_MAX_AGE = 3600  # 1 hour


def _serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def _send_reset_email(to, reset_url):
    from ..extensions import mail
    from flask_mail import Message

    if not current_app.config.get("MAIL_SERVER"):
        # No email provider configured yet - fall back to the server log so local
        # dev and early testing aren't blocked. See README for wiring a real one.
        current_app.logger.info("Password reset link for %s: %s", to, reset_url)
        return
    msg = Message("Reset your password", recipients=[to],
                  body=f"Reset your password:\n\n{reset_url}\n\n"
                       "If you didn't request this, you can ignore this email.")
    mail.send(msg)


# --------------------------------------------------------------------- signup

@auth_bp.get("/signup")
def signup_form():
    return render_template("auth/signup.html")


@auth_bp.post("/signup")
@limiter.limit("5 per minute")
def signup():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    name = (request.form.get("name") or "").strip()

    if not email or "@" not in email:
        flash("Enter a valid email address.", "error")
        return redirect(url_for("auth.signup_form"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("auth.signup_form"))
    if User.query.filter_by(email=email).first():
        flash("An account with that email already exists.", "error")
        return redirect(url_for("auth.signup_form"))

    user = User(email=email, name=name or email.split("@")[0])
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return redirect(url_for("main.app_tool"))


# ---------------------------------------------------------------------- login

@auth_bp.get("/login")
def login_form():
    return render_template("auth/login.html")


@auth_bp.post("/login")
@limiter.limit("10 per minute; 50 per day")
def login():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        flash("Incorrect email or password.", "error")
        return redirect(url_for("auth.login_form"))

    login_user(user)
    return redirect(url_for("main.app_tool"))


@auth_bp.post("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.landing"))


# --------------------------------------------------------------- google oauth

@auth_bp.get("/login/google")
def login_google():
    if not current_app.config.get("GOOGLE_CLIENT_ID"):
        flash("Google sign-in isn't configured yet.", "error")
        return redirect(url_for("auth.login_form"))
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.get("/login/google/callback")
def google_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo") or oauth.google.userinfo(token=token)
    google_id = userinfo["sub"]
    email = userinfo["email"].lower()

    user = User.query.filter_by(google_id=google_id).first()
    if user is None:
        user = User.query.filter_by(email=email).first()
        if user is None:
            user = User(email=email, name=userinfo.get("name") or email.split("@")[0])
            db.session.add(user)
        user.google_id = google_id
        db.session.commit()

    login_user(user)
    return redirect(url_for("main.app_tool"))


# --------------------------------------------------------------- password reset

@auth_bp.get("/reset")
def reset_request_form():
    return render_template("auth/reset_request.html")


@auth_bp.post("/reset")
@limiter.limit("5 per minute")
def reset_request():
    email = (request.form.get("email") or "").strip().lower()
    user = User.query.filter_by(email=email).first()
    if user:
        token = _serializer().dumps(user.email, salt=RESET_SALT)
        reset_url = url_for("auth.reset_password_form", token=token, _external=True)
        _send_reset_email(user.email, reset_url)
    # Same message whether or not the account exists, so this can't be used to
    # probe which emails are registered.
    flash("If that email has an account, a reset link has been sent.", "info")
    return redirect(url_for("auth.login_form"))


@auth_bp.get("/reset/<token>")
def reset_password_form(token):
    try:
        _serializer().loads(token, salt=RESET_SALT, max_age=RESET_MAX_AGE)
    except (BadSignature, SignatureExpired):
        flash("That reset link is invalid or has expired.", "error")
        return redirect(url_for("auth.reset_request_form"))
    return render_template("auth/reset_password.html", token=token)


@auth_bp.post("/reset/<token>")
def reset_password(token):
    try:
        email = _serializer().loads(token, salt=RESET_SALT, max_age=RESET_MAX_AGE)
    except (BadSignature, SignatureExpired):
        flash("That reset link is invalid or has expired.", "error")
        return redirect(url_for("auth.reset_request_form"))

    password = request.form.get("password") or ""
    if len(password) < 8:
        flash("Password must be at least 8 characters.", "error")
        return redirect(url_for("auth.reset_password_form", token=token))

    user = User.query.filter_by(email=email).first()
    if user:
        user.set_password(password)
        db.session.commit()

    flash("Password updated - sign in.", "info")
    return redirect(url_for("auth.login_form"))


# ------------------------------------------------------- change password (logged in)

@auth_bp.get("/change-password")
@login_required
def change_password_form():
    return render_template("auth/change_password.html")


@auth_bp.post("/change-password")
@login_required
@limiter.limit("10 per minute")
def change_password():
    new_password = request.form.get("new_password") or ""

    # Google-only accounts have no password yet - let them set one for the
    # first time without proving a "current" password that doesn't exist.
    if current_user.password_hash:
        current_password = request.form.get("current_password") or ""
        if not current_user.check_password(current_password):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("auth.change_password_form"))

    if len(new_password) < 8:
        flash("New password must be at least 8 characters.", "error")
        return redirect(url_for("auth.change_password_form"))

    current_user.set_password(new_password)
    db.session.commit()
    flash("Password updated.", "info")
    return redirect(url_for("dashboard.account"))

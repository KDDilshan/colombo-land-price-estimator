import csv
import os

from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required

import predictor

main_bp = Blueprint("main", __name__)

_RATNAPURA_PATH = os.path.join(predictor.DEPLOY, "ratnapura_points.csv")


def _load_ratnapura_points():
    """The 10 reference points from the synthetic-Ratnapura robustness check
    (see /model-notes) - environmental covariates only, no real price data,
    shown on the map for context but never priced."""
    with open(_RATNAPURA_PATH, encoding="utf-8-sig") as fh:
        return [
            {"label": row["Label"], "lat": float(row["Latitude"]), "lon": float(row["Longitude"])}
            for row in csv.DictReader(fh)
        ]


RATNAPURA_POINTS = _load_ratnapura_points()


def _accuracy():
    return {
        "median_ape": predictor.MEDIAN_APE,
        "mean_ape": predictor.MEAN_APE,
        "r2_known": predictor.R2_KNOWN,
    }


@main_bp.get("/")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for("main.app_tool"))
    return render_template(
        "landing.html",
        accuracy=_accuracy(),
        division_count=predictor.DIVISION_COUNT,
    )


@main_bp.get("/app")
@login_required
def app_tool():
    return render_template(
        "app.html",
        land_types=predictor.LAND_TYPES,
        default_land_type=predictor.DEFAULT_LAND_TYPE,
        divisions=predictor.KNOWN_DIVISIONS,
        division_count=predictor.DIVISION_COUNT,
        perch_m2=predictor.PERCH_M2,
        small_plot_perches=predictor.SMALL_PLOT_PERCHES,
        small_plot_listings=predictor.SMALL_PLOT_LISTINGS,
        accuracy=_accuracy(),
        ratnapura_points=RATNAPURA_POINTS,
    )


@main_bp.get("/terms")
def terms():
    return render_template("terms.html")


@main_bp.get("/privacy")
def privacy():
    return render_template("privacy.html")


@main_bp.get("/model-notes")
def model_notes():
    return render_template("model_notes.html", accuracy=_accuracy(), division_count=predictor.DIVISION_COUNT)

"""
step20_regenerate_thesis_figures.py
-----------------------------------
Redraws the four thesis figures whose content the 2026-09-06/07 audit changed,
reading their numbers from the artefacts rather than from hard-coded literals
wherever an artefact exists. Figures whose content did not change (Fig. 6.1
model comparison, Fig. 6.3 Gini importance) are left alone.

  figure1_1_workflow.png       seven-stage research workflow. The learner list
                               loses CatBoost, which was never executed, and
                               the evaluation boxes carry the corrected
                               same-specification results.
  fig5_1_artefact_flow.png     script-to-artefact flow. The deployment block
                               drops to six artefacts, and the ablation block
                               names the corrected scripts.
  fig6_3_spatial_vs_random.png two panels: per-fold R2 under the deterministic
                               partition against the random split, and the
                               distribution of mean R2 over 25 randomised
                               partitions, which the thesis previously could
                               not estimate.
  fig6_5_ablation.png          the attribution ladder, replacing the drop-one
                               forest plot. The drop-one result is no longer
                               the finding, so a figure organised around it
                               would mislead.

Output goes to thesis_final/ alongside the originals, which are kept as
*_PRE_AUDIT.png.
"""

from __future__ import annotations

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "thesis_final")

BG = "#FAFAF8"
BLUE = "#2077CC"
DBLUE = "#1F3B63"
RED = "#C0392B"
LRED = "#E8A8A0"
GOLD = "#B8860B"
GREEN = "#2E6B3E"
GREY = "#6B6B6B"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": "#BFBFBF",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})


def load():
    fac = json.load(open(os.path.join(PROC, "spatial_tuned_factorial.json"),
                         encoding="utf-8"))
    dist = json.load(open(os.path.join(PROC, "spatial_partition_distribution.json"),
                          encoding="utf-8"))
    ladder = pd.read_csv(os.path.join(PROC, "ablation_attribution_ladder.csv"),
                         comment="#")
    parts = pd.read_csv(os.path.join(PROC, "spatial_partition_distribution.csv"),
                        comment="#")
    return fac, dist, ladder, parts


# ------------------------------------------------------------------ flowcharts

def box(ax, x, y, w, h, title, lines, fc, ec, fs_t=13, fs_l=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=2))
    cx = x + w / 2
    n = len(lines)
    ax.text(cx, y + h - h * 0.30, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(cx, y + h * (0.55 - 0.20 * (i + 1) / max(n, 1) * n / 2.2) - 0.004 * i,
                ln, ha="center", va="center", fontsize=fs_l, zorder=3)


def simple_box(ax, x, y, w, h, title, lines, fc, ec, fs_t=13, fs_l=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=2))
    cx = x + w / 2
    rows = [(title, fs_t, "bold")] + [(l, fs_l, "normal") for l in lines]
    step = h / (len(rows) + 0.6)
    top = y + h - step * 0.85
    for i, (txt, fs, wt) in enumerate(rows):
        ax.text(cx, top - i * step, txt, ha="center", va="center",
                fontsize=fs, fontweight=wt, zorder=3)


def arrow(ax, p0, p1, color=DBLUE, lw=2.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                 linewidth=lw, color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


def fig_workflow(fac):
    fig, ax = plt.subplots(figsize=(11.5, 14.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    simple_box(ax, .13, .935, .74, .052, "STAGE 1 — DATA ACQUISITION",
               ["Ikman.lk listing scrape  |  OpenStreetMap Overpass API"], "#DCE6F5", DBLUE)
    arrow(ax, (.50, .935), (.50, .909))
    simple_box(ax, .13, .845, .74, .064, "STAGE 2 — LOCATION RESOLUTION",
               ["Colombo District → 194 Grama Niladhari divisions",
                "point-in-polygon matching, not town-name geocoding"], "#DCE6F5", DBLUE)
    ax.text(.895, .877, "replaces\nNominatim\ngeocoding", fontsize=10.5, style="italic",
            color="#7B2D26", ha="left", va="center")
    arrow(ax, (.50, .845), (.50, .818))
    simple_box(ax, .09, .742, .82, .076, "STAGE 3 — CLEANING",
               ["Global price floor and ceiling (0.5%–99.5%)  →  7,034 to 6,968",
                "Per-division trim at 1.5 SD  →  6,968 to 6,167",
                "6,167 listings across 187 divisions"], "#DCE6F5", DBLUE)

    arrow(ax, (.40, .742), (.28, .700)); arrow(ax, (.60, .742), (.72, .700))
    simple_box(ax, .03, .612, .44, .088, "STAGE 4 — AMENITY FEATURES",
               ["22 count-and-distance features",
                "across 12 facility categories",
                "exhaustive haversine scan, no spatial index"], "#DCE6F5", DBLUE)
    simple_box(ax, .53, .612, .44, .088, "STAGE 5 — HAZARD EXTRACTION",
               ["6 retained variables, 1 km buffer",
                "Copernicus DEM GLO-30, JRC GSW, MERIT Hydro,",
                "Sentinel-1 GRD, OpenStreetMap waterways"], "#FBF0CE", GOLD)
    arrow(ax, (.90, .612), (.90, .585), color=RED, lw=1.6)
    simple_box(ax, .55, .487, .44, .098, "3 CANDIDATES REJECTED",
               ["MODIS flood frequency — physical consistency",
                "JRC water recurrence — physical consistency",
                "NASA LHASA landslide (1 variable) — low variance"],
               "#F7DEDB", RED, fs_t=12, fs_l=10)

    arrow(ax, (.27, .612), (.42, .462)); arrow(ax, (.70, .612), (.58, .462))
    simple_box(ax, .16, .374, .68, .088, "MERGED DATASET",
               ["37 columns  |  194 GN divisions",
                "49 features over 6,167 listings in 187 divisions",
                "30 carried + 13 land-type dummies + 5 engineered + target encoding"],
               "#DCE6F5", DBLUE)
    arrow(ax, (.50, .374), (.50, .348))
    simple_box(ax, .13, .262, .74, .086, "STAGE 6 — MODELLING",
               ["Random Forest · Extra Trees · XGBoost · LightGBM",
                "+ equal-weight ensemble  |  RandomizedSearchCV",
                "deployed: 500-tree Random Forest, 49 features"], "#DCE6F5", DBLUE)
    arrow(ax, (.50, .262), (.50, .238))
    simple_box(ax, .27, .196, .46, .042, "ONE MODEL SPECIFICATION", [],
               "#DCE6F5", DBLUE, fs_t=12.5)
    arrow(ax, (.40, .196), (.25, .152)); arrow(ax, (.60, .196), (.75, .152))

    simple_box(ax, .02, .058, .46, .094, "STAGE 7a — RANDOM 80/20 SPLIT",
               ["4,933 train  /  1,234 test",
                f"R² = 89.43%   median APE 6.5%   mean APE 15.3%",
                "R², RMSE (log) and APE on the rupee scale"], "#DCE6F5", DBLUE, fs_t=12.5)
    simple_box(ax, .52, .058, .46, .094, "STAGE 7b — DIVISION-BLOCKED CV",
               ["GroupKFold blocked on GN division",
                "target encoding refit inside every fold",
                f"R² = 44.14%   a fall of 45.29 pp"], "#FBF0CE", GOLD, fs_t=12.5)
    ax.text(.5, .034,
            "The same model specification is evaluated under both protocols, so the gap between them "
            "reflects the partitioning scheme.",
            ha="center", va="center", fontsize=10, style="italic", color=GREY)
    ax.text(.5, .012,
            "No generated record enters the pipeline at any stage; the input file contains none.",
            ha="center", va="center", fontsize=10, style="italic", color=GREY)

    fig.savefig(os.path.join(OUT, "figure1_1_workflow.png"), dpi=160,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_artefact_flow():
    fig, ax = plt.subplots(figsize=(13.2, 15.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def lab(y, t):
        ax.text(.015, y, t, fontsize=12, fontweight="bold", color="#8A8A8A",
                ha="left", va="center")

    lab(.955, "STAGE 1")
    simple_box(ax, .12, .925, .40, .060, "01_scrape_listings.py  (archived)",
               ["raw listings CSV · title, price, perches, address"], "#DCE6F5", DBLUE, 12.5, 10)
    simple_box(ax, .57, .925, .41, .060, "step2*_fetch_*.py  (12 scripts)",
               ["data/amenities/*.csv · facility coordinates"], "#DCE6F5", DBLUE, 12.5, 10)
    arrow(ax, (.32, .925), (.32, .893)); arrow(ax, (.775, .925), (.775, .893))

    lab(.858, "STAGE 2")
    simple_box(ax, .12, .828, .40, .065, "step0 / step1a / step1b",
               ["gn_boundaries.geojson · 557 → 399 divisions",
                "listings mapped to GN division"], "#DCE6F5", DBLUE, 12.5, 10)
    simple_box(ax, .57, .828, .41, .065, "step3_clean_amenities.py",
               ["deduplicated facility lists",
                "out-of-district points removed"], "#DCE6F5", DBLUE, 12.5, 10)
    arrow(ax, (.32, .828), (.32, .795)); arrow(ax, (.775, .828), (.66, .677))

    lab(.760, "STAGE 3")
    simple_box(ax, .12, .730, .40, .065, "step1d_price_audit → step1e_clean_listings",
               ["R1–R3 drops · R4 censoring flag kept · R5 drop",
                "step1e_dropped_rows.csv (every drop + rule)"], "#FBF0CE", GOLD, 11.5, 10)
    arrow(ax, (.32, .730), (.44, .672))

    lab(.640, "STAGE 4")
    simple_box(ax, .22, .607, .56, .065, "step4_build_final_dataset.py",
               ["exhaustive haversine scan, once per division",
                "22 amenity features · two radius variants"], "#FBF0CE", GOLD, 12.5, 10)
    arrow(ax, (.42, .607), (.34, .581))

    lab(.545, "STAGE 5")
    simple_box(ax, .10, .508, .42, .070, "step6a / step6c / step6d / step6f / step6i",
               ["194 centroids · COG hazard extraction · OSM",
                "GEE JavaScript emitted for SAR and HAND"], "#FBF0CE", GOLD, 12, 10)
    simple_box(ax, .56, .508, .42, .070, "step6e_screen_and_join.py",
               ["3 screening rules · 3 rejections written to disk",
                "flood_frequency_rejection.md · landslide_null_test.md"],
               "#F7DEDB", RED, 12, 9.5)
    arrow(ax, (.36, .508), (.45, .452), color=GOLD)
    arrow(ax, (.72, .508), (.60, .452), color=RED)

    lab(.415, "MERGE")
    simple_box(ax, .20, .382, .60, .070, "final_dataset_hazard.csv",
               ["7,034 rows × 37 columns · 194 GN divisions · Colombo only",
                "cleaned to 6,167 rows across 187 divisions · no generated record"],
               "#DDEBDD", GREEN, 12.5, 10)
    arrow(ax, (.42, .382), (.30, .318)); arrow(ax, (.58, .382), (.72, .318))

    lab(.278, "STAGE 6–7")
    simple_box(ax, .06, .245, .44, .073, "random_forest_land_price_model.ipynb",
               ["random 80/20 split · RandomizedSearchCV",
                "four learners + equal-weight ensemble"], "#DCE6F5", DBLUE, 12, 10)
    simple_box(ax, .53, .245, .45, .073, "step15 / step16 / step17 / step18 / step19",
               ["division-blocked CV under the deployed specification",
                "OOF target encoding per fold · 25 randomised partitions"],
               "#DCE6F5", DBLUE, 12, 9.5)
    arrow(ax, (.30, .245), (.42, .190)); arrow(ax, (.72, .245), (.60, .190))

    lab(.150, "DEPLOY")
    simple_box(ax, .20, .110, .60, .080, "deploy/  —  six frozen serving artefacts",
               ["model.pkl · feature_order.json · division_values.csv",
                "address_encoding.csv · fallback.json · gn_boundaries.geojson"],
               "#DDEBDD", GREEN, 12.5, 10)

    lab(.055, "NOT\nARCHIVED")
    simple_box(ax, .16, .018, .68, .078, "Two artefacts cannot be re-executed",
               ["synthetic record generator — no script, seed or log survives",
                "merged two-district file build script — nothing in the repository writes it",
                "neither contributed to any result reported in this thesis"],
               "#F7DEDB", RED, 12.5, 9.5)
    ax.text(.5, -.012,
            "Every stage writes its output to disk before the next reads it; no stage holds "
            "state in memory across a run.",
            ha="center", va="center", fontsize=10.5, style="italic", color=GREY)

    fig.savefig(os.path.join(OUT, "fig5_1_artefact_flow.png"), dpi=150,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ------------------------------------------------------------------- results

def fig_spatial(fac, dist, parts):
    folds = fac["spatial_tuned_seed42_folds"]
    rnd = fac["random_tuned500_seed42_r2"]
    sp = fac["spatial_tuned500_seed42_r2"]
    gap = rnd - sp

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.6, 6.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # panel a -- per-fold under the deterministic partition
    xs = np.arange(1, 6)
    cols = [RED if v == min(folds) else BLUE for v in folds]
    a1.bar(xs, folds, width=.6, color=cols, zorder=3)
    for x, v in zip(xs, folds):
        a1.text(x, v + 1.6, f"{v:.2f}", ha="center", fontsize=11, zorder=4)
    a1.axhline(rnd, ls="--", lw=2.4, color="#7A7A7A", zorder=2)
    a1.text(5.55, rnd + 1.2, f"random split\n{rnd:.2f}%", ha="right", va="bottom",
            fontsize=11, fontweight="bold", color="#5A5A5A")
    a1.axhline(sp, lw=2.0, color=DBLUE, zorder=2)
    a1.text(5.55, sp - 5.5, f"blocked mean\n{sp:.2f}%", ha="right", va="bottom",
            fontsize=11, fontweight="bold", color=DBLUE)
    a1.annotate("", xy=(0.62, rnd), xytext=(0.62, sp),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="#333333"))
    a1.text(0.78, (rnd + sp) / 2, f"{gap:.2f} pp", rotation=90, ha="center",
            va="center", fontsize=13, fontweight="bold")
    a1.set_xticks(xs, [f"Fold {i}" for i in xs], fontsize=11.5)
    a1.set_ylim(0, 100); a1.set_xlim(0.35, 5.75)
    a1.set_ylabel("R² (%)", fontsize=12.5)
    a1.grid(axis="y", color="#DDDDDD", zorder=0)
    a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.set_title("(a)  Per-fold R² under the deterministic GroupKFold partition,\n"
                 "deployed 500-tree specification, seed 42",
                 fontsize=12, loc="left", pad=12)

    # panel b -- distribution over randomised partitions
    r = parts[parts.kind == "randomised"]["mean_r2"].to_numpy()
    a2.hist(r, bins=9, color=BLUE, alpha=.78, zorder=3, edgecolor="white")
    a2.axvline(float(np.mean(r)), lw=2.2, color=DBLUE, zorder=4)
    a2.text(float(np.mean(r)) - 0.6, a2.get_ylim()[1] * .92,
            f"mean {np.mean(r):.2f}%\nSD {np.std(r, ddof=1):.2f}", ha="right",
            fontsize=11, color=DBLUE, fontweight="bold")
    a2.axvline(dist["deterministic_groupkfold_r2"], lw=2.4, color=RED, zorder=4)
    a2.text(dist["deterministic_groupkfold_r2"] + 0.4, a2.get_ylim()[1] * .55,
            f"deterministic\nGroupKFold\n{dist['deterministic_groupkfold_r2']:.2f}%",
            ha="left", fontsize=11, color=RED, fontweight="bold")
    a2.set_xlabel("Mean R² (%) over five division-blocked folds", fontsize=12)
    a2.set_ylabel("Number of partitions", fontsize=12)
    a2.grid(axis="y", color="#DDDDDD", zorder=0)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    a2.set_title("(b)  25 independent division-to-fold assignments.\n"
                 "The deterministic partition is the most favourable of the 26.",
                 fontsize=12, loc="left", pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_3_spatial_vs_random.png"), dpi=160,
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def fig_ladder(ladder):
    lab = ["1.  untuned 300-tree\n     five-variable block\n     fixed partition",
           "2.  deployed 500-tree\n     five-variable block\n     fixed partition",
           "3.  deployed 500-tree\n     six-variable block\n     fixed partition",
           "4.  deployed 500-tree\n     six-variable block\n     randomised partitions"]
    d = ladder["delta_pp"].to_numpy()
    e = ladder["two_sd_pp"].to_numpy()
    meets = ladder["meets_decision_rule"].astype(str).str.lower().eq("true").to_numpy()

    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    ys = np.arange(len(d))[::-1]
    for y, v, err, m in zip(ys, d, e, meets):
        c = RED if m else "#8FA9C4"
        ax.errorbar(v, y, xerr=err, fmt="o", color=c, ecolor=c,
                    elinewidth=2.6 if m else 2.0, capsize=5,
                    markersize=11 if m else 8, zorder=3)
        ax.text(v, y + .27, f"{v:+.2f} pp", ha="center", fontsize=12,
                fontweight="bold" if m else "normal")
        ax.text(4.75, y, "meets the rule" if m else "does not meet the rule",
                ha="right", va="center", fontsize=11,
                color=RED if m else GREY, fontweight="bold" if m else "normal")
    ax.axvline(0, color="#8A8A8A", lw=1.6, zorder=1)
    ax.set_yticks(ys, lab, fontsize=11)
    ax.set_xlim(-3.6, 4.9)
    ax.set_ylim(-0.6, len(d) + 0.25)
    ax.set_xlabel("Change in division-blocked R² when the whole hazard block is removed "
                  "(percentage points)", fontsize=12.5)
    ax.grid(axis="x", color="#E4E4E4", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(-3.55, len(d) + 0.18,
            "Bars are ±2 SD of the paired difference. Each rung differs from the one above it "
            "in exactly one respect,\nso the change between adjacent rungs is attributable to "
            "that change alone.",
            fontsize=11, style="italic", color="#555555", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_5_ablation.png"), dpi=160,
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    fac, dist, ladder, parts = load()
    for f in ("figure1_1_workflow.png", "fig5_1_artefact_flow.png",
              "fig6_3_spatial_vs_random.png", "fig6_5_ablation.png"):
        src = os.path.join(OUT, f)
        dst = os.path.join(OUT, f.replace(".png", "_PRE_AUDIT.png"))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    fig_workflow(fac)
    fig_artefact_flow()
    fig_spatial(fac, dist, parts)
    fig_ladder(ladder)
    for f in ("figure1_1_workflow.png", "fig5_1_artefact_flow.png",
              "fig6_3_spatial_vs_random.png", "fig6_5_ablation.png"):
        p = os.path.join(OUT, f)
        print(f"  wrote {f:34s} {os.path.getsize(p):>8,} bytes")


if __name__ == "__main__":
    main()

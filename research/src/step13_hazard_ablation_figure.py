"""Step 13 - the hazard ablation figure.

Step 7 computed the ablation and wrote three CSVs. It never drew it, so the
central result of the thesis - that the hazard block buys real spatial
generalisation - had no figure. This script fills that gap.

Input  data/processed/hazard_ablation_summary.csv   (written by step7)
Output results/figures/fig9_hazard_ablation.png     300 dpi
       results/figures/fig9_hazard_ablation.csv     the plotted values

WHICH ABLATION NUMBER THIS PLOTS - read before quoting it
  Two ablation results exist and they disagree:
    model test/hazard_ablation_results.csv    single seed, no_hazard -4.07 pp
    data/processed/hazard_ablation_summary.csv  5 seeds,   no_hazard -3.06 pp
  This figure plots the MULTISEED file. GroupKFold is deterministic, so the
  folds are identical across seeds; the seed varies only the RandomForest and
  the target encoder. The single-seed -4.07 pp is one draw from a distribution
  whose SD is 1.49 pp. Quote -3.06 pp and show the spread.

THE NOISE FLOOR
  step7 marked a configuration `above_noise_floor` when its delta exceeds the
  seed-to-seed variation of the baseline itself. Only two clear it: removing
  the whole hazard block, and removing water_occurrence_pct. The other four
  single-variable drops sit inside the noise, and two of them are POSITIVE -
  the model scores marginally better without dist_to_stream_km or
  sar_flood_open_land. That is plotted as it stands, not hidden. The honest
  reading is that the hazard block works as a block, carried mainly by
  water_occurrence_pct, and the remaining variables are not individually
  separable at n=194 divisions.

PANEL A IS NOT A BAR CHART, DELIBERATELY
  Mean R2 spans 48-52%, so bars from zero would flatten the differences and
  bars from a cut axis would exaggerate them. Points with seed-range whiskers
  state the same values without either distortion. Panel B is a difference
  from a baseline, so bars from zero are correct there.
"""

from __future__ import annotations

import datetime as dt
import os
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "processed", "hazard_ablation_summary.csv")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

DPI = 300
STARTED = dt.datetime.now()
ENV = (f"python={platform.python_version()} numpy={np.__version__} "
       f"pandas={pd.__version__} matplotlib={matplotlib.__version__}")

LABEL = {
    "baseline_all5_hazard":      "All five hazard variables\n(baseline)",
    "no_hazard":                 "Hazard block removed\nentirely",
    "drop_water_occurrence_pct": "− water_occurrence_pct",
    "drop_hand_m":               "− hand_m",
    "drop_dist_to_kelani_km":    "− dist_to_kelani_km",
    "drop_sar_flood_open_land":  "− sar_flood_open_land",
    "drop_dist_to_stream_km":    "− dist_to_stream_km",
}

C_SIG = "#C44E52"      # clears the noise floor
C_NS = "#B0B0B0"       # within seed noise
C_BASE = "#4C72B0"     # baseline
C_REF = "#333333"

# ---------------------------------------------------------------- load
with open(SRC, encoding="utf-8") as fh:
    header_lines = [ln for ln in fh if ln.startswith("#")]
header = " ".join(header_lines)

def _meta(key, cast=str):
    for tok in header.replace("#", " ").split():
        if tok.startswith(key + "="):
            return cast(tok.split("=", 1)[1])
    raise ValueError(f"{key} not found in {SRC} header")

rows_used = _meta("rows_used", int)
n_groups = _meta("groups", int)

df = pd.read_csv(SRC, comment="#")
assert len(df) == 7, len(df)

df["above_noise_floor"] = df["above_noise_floor"].fillna(False).astype(bool)
df["is_base"] = df["config"] == "baseline_all5_hazard"
base_r2 = float(df.loc[df["is_base"], "mean_r2"].iloc[0])

# shared ordering: most harmful removal at the top
df = df.sort_values("delta_vs_baseline_pp", ascending=False).reset_index(drop=True)
y = np.arange(len(df))
labels = [LABEL[c] for c in df["config"]]

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(13.5, 6.0), sharey=True,
    gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.08})

# ------------------------------------------------------- PANEL A: mean R2
for i, r in df.iterrows():
    colour = C_BASE if r["is_base"] else (C_SIG if r["above_noise_floor"] else C_NS)
    axA.plot([r["min_seed_mean"], r["max_seed_mean"]], [i, i],
             color=colour, lw=2.0, alpha=0.55, solid_capstyle="round", zorder=2)
    axA.plot(r["mean_r2"], i, "o", ms=9, color=colour,
             markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    axA.text(r["max_seed_mean"] + 0.28, i, f"{r['mean_r2']:.2f}",
             va="center", ha="left", fontsize=9, color=C_REF)

axA.axvline(base_r2, color=C_BASE, ls="--", lw=1.2, alpha=0.8, zorder=1,
            label=f"baseline {base_r2:.2f}%")
axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=10)
axA.set_xlabel("Spatial-CV R² (%), mean of 5 seeds")
axA.set_title("A.  Model skill by hazard configuration\n"
              "point = mean, bar = range across seeds", fontsize=11)
axA.set_xlim(46.0, 56.2)
axA.legend(loc="lower right", fontsize=9, frameon=False)
axA.grid(axis="x", alpha=0.25, lw=0.6)
axA.set_axisbelow(True)

# ------------------------------------------------- PANEL B: delta vs baseline
plot = df[~df["is_base"]]
for i, r in plot.iterrows():
    colour = C_SIG if r["above_noise_floor"] else C_NS
    axB.barh(i, r["delta_vs_baseline_pp"], height=0.55, color=colour,
             alpha=0.95 if r["above_noise_floor"] else 0.75, zorder=2)
    axB.errorbar(r["delta_vs_baseline_pp"], i, xerr=r["delta_sd_pp"],
                 fmt="none", ecolor=C_REF, elinewidth=1.1, capsize=3.5,
                 alpha=0.75, zorder=3)
    # park the value clear of the whisker cap on whichever side the bar runs
    d, sd = r["delta_vs_baseline_pp"], r["delta_sd_pp"]
    x = d - sd - 0.25 if d < 0 else d + sd + 0.25
    axB.text(x, i, f"{d:+.2f}", va="center",
             ha="right" if d < 0 else "left", fontsize=9.5,
             fontweight="bold" if r["above_noise_floor"] else "normal",
             color=C_SIG if r["above_noise_floor"] else C_REF)
    axB.text(4.75, i, f"{int(r['delta_seeds_same_sign'])}/5",
             va="center", ha="center", fontsize=8.5, color="#666666")

axB.axvline(0, color=C_BASE, ls="--", lw=1.2, alpha=0.8, zorder=1)
axB.text(4.75, len(df) - 0.42, "seeds\nagreeing", ha="center", va="center",
         fontsize=8, color="#666666")
axB.set_xlabel("Change in spatial-CV R² vs baseline (percentage points)")
axB.set_title("B.  Cost of removing each hazard variable\n"
              "bar = mean Δ, whisker = ±1 SD across seeds", fontsize=11)
axB.set_xlim(-5.7, 5.6)
axB.set_ylim(-0.75, len(df) - 0.12)
axB.grid(axis="x", alpha=0.25, lw=0.6)
axB.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=C_SIG, alpha=0.95),
           plt.Rectangle((0, 0), 1, 1, color=C_NS, alpha=0.75)]
axB.legend(handles, ["clears the seed-noise floor", "within seed noise"],
           loc="lower left", fontsize=9, frameon=False)

no_hazard_gain = -float(df.loc[df["config"] == "no_hazard", "delta_vs_baseline_pp"].iloc[0])

fig.suptitle(
    f"Hazard ablation: the flood-hazard block buys {no_hazard_gain:.2f} pp of "
    "spatial-CV R², carried mainly by water_occurrence_pct",
    fontsize=12.5, y=0.985)

fig.text(0.5, -0.045,
         "RandomForest(300 trees) - GroupKFold(5) by Address (unseen GN division) - "
         "target log(price/perch), out-of-fold target encoding refit per fold\n"
         f"{rows_used:,} listings after price floor/ceiling + per-Address ±1.5 SD "
         f"trim, {n_groups} divisions - "
         "seeds 42/1/7/123/2024 - elevation_m excluded (r = 0.93 with hand_m)\n"
         "Source: data/processed/hazard_ablation_summary.csv (step7). Negative = "
         "the model got worse without that variable.",
         ha="center", va="top", fontsize=8.2, color="#555555")

out_png = os.path.join(FIGDIR, "fig9_hazard_ablation.png")
fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  wrote results/figures/{os.path.basename(out_png)}")

# ---------------------------------------------------------------- plotted CSV
cols = ["config", "mean_r2", "sd_across_seeds", "min_seed_mean", "max_seed_mean",
        "delta_vs_baseline_pp", "delta_sd_pp", "delta_seeds_same_sign",
        "above_noise_floor"]
out_csv = os.path.join(FIGDIR, "fig9_hazard_ablation.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as fh:
    fh.write(
        f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step13_hazard_ablation_figure.py\n"
        f"# env {ENV}\n"
        "# source=data/processed/hazard_ablation_summary.csv (step7, 5 seeds)\n"
        "# these are the values plotted in fig9_hazard_ablation.png, re-ordered\n"
        "#   by delta; no recomputation was performed by this script\n"
        "# NOT the single-seed model test/hazard_ablation_results.csv (-4.07 pp)\n")
    df[cols].to_csv(fh, index=False)
print(f"  wrote results/figures/{os.path.basename(out_csv)}")

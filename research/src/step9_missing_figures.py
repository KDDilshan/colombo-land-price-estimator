"""Step 9 - the three figures flagged [MISSING] in Chapter 4.

Chapter 4 records three absences that this script fills:
  4.2  no histogram or density plot of the price distribution
  4.2  no correlation-matrix figure for the hazard variables
  4.2  no map of the 194 divisions, their hazard values or the price surface

Outputs (results/figures/, 300 dpi PNG, each with a CSV of the plotted values)
  fig6_price_distribution.png      raw and log price per perch
  fig7_hazard_correlation.png      6x6 Pearson matrix, divisional and row basis
  fig8_gn_division_map.png         choropleth, median price and water_occurrence_pct
  fig6_price_distribution.csv      bin edges and counts
  fig7_hazard_correlation.csv      the plotted correlation matrices
  fig8_gn_division_map.csv         per-division values plotted

BASIS NOTE - this matters for the correlation figure
  Hazard columns are constant within a GN division, so a correlation computed
  across the 7,034 listing rows weights each division by its listing count
  (median 7, max 543). The divisional basis (n=194) is the basis on which the
  variables were screened (hazard_screening.md) and is the primary panel here.
  The row basis is plotted alongside because it is what the model sees. The
  elevation_m / hand_m pair differs sharply between them: +0.927 divisional
  against +0.753 row-wise. Both are correct.

MAP CAVEAT - read before using fig8
  `Address` is a bare GN division name with no Divisional Secretariat
  qualifier (Chapter 3, Section 3.3.2). The polygon file holds 557 records and
  193 of the 194 analysis units match MORE THAN ONE polygon, because GN names
  repeat across sub-divisions ("Kottawa" matches 5). Each analysis unit is
  therefore drawn as the union of every polygon sharing its base name. The map
  is a faithful picture of the ANALYSIS UNIT, not of a single administrative
  polygon, and it makes the bare-name limitation visible rather than hiding it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "final_dataset_hazard.csv")
CENTROIDS = os.path.join(ROOT, "data", "processed", "gn_centroids.csv")
POLYS = os.path.join(ROOT, "data", "reference", "colombo_adm4_polygons.json")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

DPI = 300
STARTED = dt.datetime.now()
ENV = (f"python={platform.python_version()} numpy={np.__version__} "
       f"pandas={pd.__version__} matplotlib={matplotlib.__version__}")

HAZ6 = ["elevation_m", "water_occurrence_pct", "dist_to_stream_km",
        "dist_to_kelani_km", "hand_m", "sar_flood_open_land"]


def header(extra):
    lines = [f"# generated {STARTED:%Y-%m-%d %H:%M} by src/step9_missing_figures.py",
             "# dataset=data/processed/final_dataset_hazard.csv (7034 rows x 37 cols)",
             f"# env {ENV}",
             "# these are DESCRIPTIVE statistics of the dataset, not model output;",
             "#   they are exact and do not depend on library version"]
    return "\n".join(lines + [f"# {e}" for e in extra]) + "\n"


def write_csv(path, df, extra):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(header(extra))
        df.to_csv(fh, index=False)
    print(f"  wrote results/figures/{os.path.basename(path)}")


def savefig(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


df = pd.read_csv(DATA, low_memory=False)
assert df.shape == (7034, 37), df.shape
div = df.groupby("Address")[HAZ6].first()
med_price = df.groupby("Address")["Price per Perch"].median()
n_list = df.groupby("Address").size()

# ------------------------------------------------------------------ FIGURE 6
print("[1] price distribution")
ppp = df["Price per Perch"].values
lppp = np.log(ppp)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(ppp / 1e6, bins=60, color="#4C72B0", alpha=0.85)
axes[0].axvline(np.median(ppp) / 1e6, color="black", ls="--", lw=1.2,
                label=f"median {np.median(ppp)/1e6:.2f}M")
axes[0].axvline(ppp.mean() / 1e6, color="#C44E52", ls=":", lw=1.4,
                label=f"mean {ppp.mean()/1e6:.2f}M")
axes[0].set_xlabel("Price per perch (Rs, millions)")
axes[0].set_ylabel("Listings")
axes[0].set_title(f"Raw price per perch (skew {pd.Series(ppp).skew():.2f})")
axes[0].legend(fontsize=8)

axes[1].hist(lppp, bins=60, color="#55A868", alpha=0.85)
axes[1].axvline(np.median(lppp), color="black", ls="--", lw=1.2,
                label=f"median {np.median(lppp):.2f}")
axes[1].set_xlabel("log(price per perch)")
axes[1].set_ylabel("Listings")
axes[1].set_title(f"Log-transformed target (skew {pd.Series(lppp).skew():.3f})")
axes[1].legend(fontsize=8)
fig.suptitle("Distribution of the modelling target, 7,034 listings", y=1.02)
fig.tight_layout()
savefig(fig, "fig6_price_distribution.png")

c_raw, e_raw = np.histogram(ppp, bins=60)
c_log, e_log = np.histogram(lppp, bins=60)
bins = pd.DataFrame({
    "basis": ["raw"] * 60 + ["log"] * 60,
    "bin_left": list(e_raw[:-1]) + list(e_log[:-1]),
    "bin_right": list(e_raw[1:]) + list(e_log[1:]),
    "count": list(c_raw) + list(c_log)})
n_ceiling = int((ppp >= 1e7).sum())
write_csv(os.path.join(FIGDIR, "fig6_price_distribution.csv"), bins,
          [f"raw: min={ppp.min():.0f} median={np.median(ppp):.0f} mean={ppp.mean():.0f} "
           f"max={ppp.max():.0f} sd={ppp.std(ddof=1):.0f} skew={pd.Series(ppp).skew():.4f}",
           f"log: min={lppp.min():.4f} median={np.median(lppp):.4f} mean={lppp.mean():.4f} "
           f"max={lppp.max():.4f} sd={lppp.std(ddof=1):.4f} skew={pd.Series(lppp).skew():.4f}",
           f"CEILING: {n_ceiling} rows sit exactly at Rs 10,000,000/perch - the target is",
           "  right-censored; the Colombo 1/2/3/7 core is not represented"])

# ------------------------------------------------------------------ FIGURE 7
print("[2] hazard correlation matrix")
c_div = div[HAZ6].corr()
c_row = df[HAZ6].corr()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, mat, title, n in ((axes[0], c_div, "Divisional basis (screening basis)", 194),
                          (axes[1], c_row, "Listing-row basis (what the model sees)", 7034)):
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(HAZ6)))
    ax.set_yticks(range(len(HAZ6)))
    ax.set_xticklabels(HAZ6, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(HAZ6, fontsize=8)
    ax.set_title(f"{title}\nn = {n:,}", fontsize=10)
    for i in range(len(HAZ6)):
        for j in range(len(HAZ6)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black",
                    fontweight="bold" if (abs(v) > 0.8 and i != j) else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
fig.suptitle("Pearson correlation among the six extracted hazard variables\n"
             "bold = above the |r| = 0.8 screening threshold", y=1.04)
fig.tight_layout()
savefig(fig, "fig7_hazard_correlation.png")

rows = []
for basis, mat in (("divisional_194", c_div), ("listing_rows_7034", c_row)):
    for a in HAZ6:
        for b in HAZ6:
            rows.append({"basis": basis, "var_a": a, "var_b": b,
                         "pearson_r": round(float(mat.loc[a, b]), 6)})
write_csv(os.path.join(FIGDIR, "fig7_hazard_correlation.csv"), pd.DataFrame(rows),
          [f"elevation_m vs hand_m: divisional {c_div.loc['elevation_m','hand_m']:.4f}, "
           f"row {c_row.loc['elevation_m','hand_m']:.4f}",
           "the divisional figure breaches the 0.80 screening threshold and the row",
           "  figure does not; elevation_m was excluded on the divisional basis",
           "  (hazard_screening.md 3c). Both values are correct."])

# ------------------------------------------------------------------ FIGURE 8
print("[3] GN division map")
cen = pd.read_csv(CENTROIDS)
polys = json.load(open(POLYS, encoding="utf-8"))


def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


# Index on BOTH base_name and full name. One analysis unit, 'havelock town',
# matches the polygon's `name` ("Havelock Town") but not its `base_name`
# ("Havelock"), so indexing on base_name alone loses it. Where a unit matches
# under both keys, base_name wins - it is the coarser and intended join key.
bybase, byname = {}, {}
for f in polys:
    bybase.setdefault(norm(f.get("base_name")), []).append(f)
    byname.setdefault(norm(f.get("name")), []).append(f)


def lookup(unit):
    k = norm(unit)
    return bybase.get(k) or byname.get(k) or []

units = sorted(div.index)
matched, n_poly, ds_span = 0, 0, 0
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
panels = [("Median price per perch (Rs millions)", med_price / 1e6, "viridis"),
          ("water_occurrence_pct", div["water_occurrence_pct"], "Blues")]

plotted = []
for ax, (title, series, cmap) in zip(axes, panels):
    vals = series.reindex(units).values
    vmin, vmax = np.nanmin(vals), np.nanpercentile(vals, 97)
    norm_f = plt.Normalize(vmin=vmin, vmax=vmax)
    cm = matplotlib.colormaps[cmap]
    patches, colors = [], []
    for u, v in zip(units, vals):
        feats = lookup(u)
        for f in feats:
            for ring in f["rings"]:
                patches.append(MplPolygon(np.asarray(ring), closed=True))
                colors.append(cm(norm_f(v)))
    pc = PatchCollection(patches, facecolors=colors, edgecolors="white", linewidths=0.15)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{len(units)} analysis units, {len(patches)} polygons", fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_f, cmap=cm), ax=ax,
                 fraction=0.04, label=title)

for u in units:
    feats = lookup(u)
    if feats:
        matched += 1
        n_poly += len(feats)
        if len({f.get("ds") for f in feats}) > 1:
            ds_span += 1
    plotted.append({"Address": u,
                    "n_polygons": len(feats),
                    "ds_divisions": "|".join(sorted({str(f.get("ds")) for f in feats})),
                    "n_listings": int(n_list.get(u, 0)),
                    "median_price_per_perch": float(med_price.get(u, np.nan)),
                    "water_occurrence_pct": float(div["water_occurrence_pct"].get(u, np.nan)),
                    "hand_m": float(div["hand_m"].get(u, np.nan)),
                    "sar_flood_open_land": float(div["sar_flood_open_land"].get(u, np.nan)),
                    "gn_lat": float(cen.set_index("Address")["gn_lat"].get(u, np.nan)),
                    "gn_lon": float(cen.set_index("Address")["gn_lon"].get(u, np.nan))})

fig.suptitle("Colombo District GN divisions carrying listings: price and surface-water exposure\n"
             "colour scales clipped at the 97th percentile", y=1.0)
fig.tight_layout()
savefig(fig, "fig8_gn_division_map.png")

print(f"  matched {matched}/{len(units)} units to {n_poly} polygons; "
      f"{ds_span} units span more than one DS division")
write_csv(os.path.join(FIGDIR, "fig8_gn_division_map.csv"), pd.DataFrame(plotted),
          [f"matched {matched} of {len(units)} analysis units to {n_poly} polygons",
           f"{ds_span} units resolve to polygons in MORE THAN ONE DS division - direct",
           "  evidence of the bare-GN-name limitation in Chapter 3 Section 3.3.2",
           "colour scales clipped at the 97th percentile to keep outliers from",
           "  flattening the ramp; see the CSV for unclipped values",
           "median_price_per_perch is the per-division median over its listings"])

print(f"\n[DONE] three figures + three value CSVs in "
      f"{(dt.datetime.now()-STARTED).total_seconds():.0f}s")

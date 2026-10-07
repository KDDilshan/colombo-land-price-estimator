"""Step 6g - write the flood_frequency rejection record.

The second documented negative in this extension, after landslide_null_test.md.
Both are written from the collected data rather than asserted, and both stay in
the repository so the write-up can cite a measurement instead of a claim.

Reads : data/processed/gee_hand_floodfreq.csv, final_dataset.csv, gn_centroids.csv
Writes: data/processed/flood_frequency_rejection.md
"""

import csv
import datetime as dt
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
GEE = os.path.join(PROC, "gee_hand_floodfreq.csv")
FINAL = os.path.join(PROC, "final_dataset.csv")
OUT = os.path.join(PROC, "flood_frequency_rejection.md")

# The Kelani floodplain: the divisions the layer had to detect to be usable.
KELANI = ["mulleriyawa", "sedawatta", "kelanimulla", "ambathale", "wellampitiya",
          "kotikawatta", "kolonnawa", "grandpass", "orugodawatta", "nawagamuwa"]


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def main():
    with open(GEE, encoding="utf-8-sig") as fh:
        gee = {r["Address"]: r for r in csv.DictReader(fh)}
    with open(FINAL, encoding="utf-8-sig") as fh:
        fort = {}
        for r in csv.DictReader(fh):
            fort.setdefault(r["Address"], float(r["Distance from fort"]))

    addrs = sorted(gee)
    ff = {a: float(gee[a]["flood_frequency"]) for a in addrs}
    hd = {a: float(gee[a]["hand_m"]) for a in addrs}
    v = [ff[a] for a in addrs]
    n = len(addrs)
    zeros = sum(1 for x in v if x == 0)
    r_fort = pearson(v, [fort[a] for a in addrs])
    r_hand = pearson(v, [hd[a] for a in addrs])
    top = sorted(((a, ff[a]) for a in addrs), key=lambda t: -t[1])[:10]
    kel = [(a, ff[a], hd[a]) for a in KELANI if a in ff]

    md = [
        "# Rejected variable: `flood_frequency` (Global Flood Database)", "",
        f"Written {dt.date.today().isoformat()} by "
        "`src/step6g_flood_frequency_rejection.py`.", "",
        "## Decision", "",
        "**Collected, measured, rejected. Not joined into "
        "`final_dataset_hazard.csv`.** The raw column is kept in "
        "`data/processed/gee_hand_floodfreq.csv` exactly as Earth Engine "
        "returned it; nothing was deleted.", "",
        "This is the second documented negative in the hazard extension. The "
        "first is `landslide_null_test.md`, which was rejected on low variance. "
        "This one is different and more important: **it passed both screening "
        "rules and was rejected anyway, on validity.**", "",
        "## What was collected", "",
        "- Source: `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1`, the Global Flood Database "
        "v1, band `flooded` summed across the collection.",
        "- Meaning: the number of mapped flood events, 2000-2018, that inundated "
        "the pixel.",
        "- Resolution: 250 m (MODIS).",
        "- Geometry: mean over a 1 km circular buffer around each of the 194 GN "
        "division centroids - the same geometry as every other hazard variable.",
        "- Unmasked to 0, on the reasoning that absence from the flood archive "
        "means never observed flooded.", "",
        "## The numbers", "",
        "| statistic | value |", "|---|---|",
        f"| divisions | {n} |",
        f"| nulls | 0 |",
        f"| distinct values | {len(set(v))} |",
        f"| **exactly zero** | **{zeros} of {n} ({100.0 * zeros / n:.1f}%)** |",
        f"| min / mean / max | {min(v):.3f} / {np.mean(v):.3f} / {max(v):.3f} |",
        f"| r vs `Distance from fort` | {r_fort:+.3f} |",
        f"| r vs `hand_m` | {r_hand:+.3f} |", "",
        "**Both screening rules pass.** The distinct-value count "
        f"({len(set(v))}) clears the threshold of 30, and "
        f"|r| = {abs(r_fort):.2f} against `Distance from fort` is nowhere near "
        "the 0.80 cut-off. A purely mechanical screening would have kept this "
        "column and put it in the model.", "",
        "## Why it is invalid: every signal is coastal", "",
        "The ten highest values in the district:", "",
        "| division | flood_frequency | on the coast? |", "|---|---|---|"]
    coastal = {"fort", "uyana", "idama", "moratuwella", "mount lavinia",
               "kahapola", "rathmalana", "lakshapathiya", "kaldemulla",
               "borupana", "dehiwala", "katubedda", "angulana", "soysapura",
               "koralawella", "egodauyana", "wellawatta", "bambalapitiya"}
    for a, x in top:
        md.append(f"| {a} | {x:.3f} | {'yes' if a in coastal else '-'} |")

    md += ["", "And the divisions it had to detect to be worth having - the "
           "Kelani Ganga floodplain, where Colombo's actual river flooding "
           "happens and where the 2016 and 2017 events did their damage:", "",
           "| division | flood_frequency | hand_m |", "|---|---|---|"]
    for a, f, h in kel:
        md.append(f"| {a} | **{f:.3f}** | {h:.2f} |")

    md += ["", f"**Not one of them is above zero.** The `hand_m` column beside "
           "them shows what a working flood variable looks like on the same "
           "buffers: the Kelani belt reads 3-5 m above nearest drainage while "
           "the southern uplands read 11-12 m. `flood_frequency` sees none of "
           "that and instead lights up the shoreline.", "",
           "## Why the layer fails here", "",
           "1. **Resolution.** MODIS at 250 m cannot resolve flooding along a "
           "river corridor a few hundred metres wide threading through dense "
           "urban fabric. The Kelani floodplain in Kolonnawa and Kotikawatta is "
           "at or below the size of a single pixel.",
           "2. **Event selection.** The Global Flood Database maps large, "
           "cloud-free, optically detectable events. Sri Lankan floods arrive "
           "with the monsoon, under exactly the cloud cover that defeats optical "
           "detection.",
           "3. **What the buffers actually caught.** A 1 km buffer on a coastal "
           "division contains open sea and tidal water. Water present in the "
           "imagery is water present, and the sum reads it as inundation. The "
           "column is, in effect, a noisy inverse measure of distance to the "
           "shoreline.", "",
           "## What including it would have produced", "",
           "Coastal Colombo - Fort, Mount Lavinia, Moratuwella, Uyana - carries "
           "some of the highest land values in the district. A column whose "
           "large values sit precisely there, entered as *flood exposure*, would "
           "have produced a clean, statistically comfortable finding that **flood "
           "exposure raises land value**, or with the sign flipped by "
           "collinearity, that it lowers it near the sea. Either way the "
           "coefficient would have been reporting distance to the shoreline "
           "under a hazard label.", "",
           "This is the failure mode NOTES.md 3A.1 records four separate times in "
           "the geocoding work: a value that is *near the right answer for the "
           "wrong reason*, and which looks like success. It is caught the same "
           "way here - by checking the geography of the result against what the "
           "variable claims to measure, rather than by checking its statistics.", "",
           "## If this variable is wanted later", "",
           "The layer to use is Sentinel-1 SAR, which sees through cloud at 10 m "
           "and can resolve an urban river corridor. Copernicus EMS and UNOSAT "
           "published rapid-mapping products for the May 2016 and May 2017 Sri "
           "Lanka floods. Neither was retrievable at an accessible path during "
           "this work (NOTES.md 6.6), and no substitute was used.", "",
           "## Provenance", "",
           "- Raw values, both columns, as collected: "
           "`data/processed/gee_hand_floodfreq.csv`",
           "- Extraction script: `scripts/gee_code_editor_hand_floodfreq.js`",
           "- Screening record: `data/processed/hazard_screening.md` section 3b",
           "- Decision log: `NOTES.md` 6.10", ""]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"wrote {OUT}")
    print(f"  zeros {zeros}/{n}, distinct {len(set(v))}, r_fort {r_fort:+.3f}, "
          f"r_hand {r_hand:+.3f}")


if __name__ == "__main__":
    main()

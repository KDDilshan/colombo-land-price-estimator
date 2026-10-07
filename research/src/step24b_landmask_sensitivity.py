"""Step 24b - sensitivity check on the Rule 3a outcome: recompute the three
Landsat variables with water pixels excluded from each buffer.

The screened variables are unmasked buffer means (the method of the existing
six). Coastal buffers are up to 38% sea, and water has negative NDVI and a
cool daytime surface, so the sea can move a division's value - and with it
the correlation against Distance from fort that Rule 3a tests. This asks
whether the NDVI pass / LST fail / NDBI boundary depend on that.

Land mask: pixels whose median QA water flag over 2023-2025 is 0.
Reads the step23b cache; does NOT change the screening outcome, which stays
on the pre-specified method. Report only.

Output: data/processed/env9_landmask_sensitivity.csv (+ printed table)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step23b_extract_landsat_climate as s23        # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")


def main():
    names, xs, ys = s23.load_points()
    tr, w, h = s23.grid(xs, ys)
    per, union = s23.buffer_index(tr, w, h, xs, ys)
    mm = {k: np.load(os.path.join(s23.CACHE, f"{k}.npy"), mmap_mode="r")
          for k in ("ndvi", "ndbi", "lst", "water")}
    p = len(union)
    med = {k: np.full(p, np.nan, np.float32) for k in ("ndvi", "ndbi", "lst", "water")}
    for s in range(0, p, 20000):
        e = min(s + 20000, p)
        for k, scale, nod in (("ndvi", 1e4, s23.NODATA), ("ndbi", 1e4, s23.NODATA),
                              ("lst", 1e2, s23.NODATA), ("water", 1, -1)):
            blk = np.asarray(mm[k][:, s:e]).astype(np.float32)
            blk[blk == nod] = np.nan
            with np.errstate(all="ignore"):
                med[k][s:e] = np.nanmedian(blk, axis=0) / scale
    land = med["water"] < 0.5
    pos = {v: i for i, v in enumerate(union)}
    rows = []
    for a, idx in zip(names, per):
        j = np.fromiter((pos[v] for v in idx), np.int64, len(idx))
        jl = j[land[j]]
        rows.append({"Address": a, "land_share": len(jl) / len(j),
                     **{f"{k}_land": float(np.nanmean(med[k][jl])) for k in ("ndvi", "ndbi", "lst")}})
    lm = pd.DataFrame(rows).rename(columns={"lst_land": "lst_day_c_land"})

    base = pd.read_csv(os.path.join(PROC, "final_dataset_env9.csv"), low_memory=False)
    d = base.groupby("Address").first()[["Distance from fort", "lst_day_c", "ndvi", "ndbi",
                                         "water_occurrence_pct"]]
    d = d.join(lm.set_index("Address"))
    out = []
    for v in ("lst_day_c", "ndvi", "ndbi"):
        out.append(dict(variable=v,
                        r_fort_unmasked=d[v].corr(d["Distance from fort"]),
                        r_fort_land_only=d[f"{v}_land"].corr(d["Distance from fort"]),
                        r_water_occ_unmasked=d[v].corr(d["water_occurrence_pct"]),
                        r_water_occ_land_only=d[f"{v}_land"].corr(d["water_occurrence_pct"]),
                        r_unmasked_vs_land=d[v].corr(d[f"{v}_land"])))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(PROC, "env9_landmask_sensitivity.csv"), index=False,
               float_format="%.4f")
    lm.to_csv(os.path.join(PROC, "landsat_climate_194_landmasked.csv"), index=False,
              float_format="%.5f")
    print(out.to_string(index=False, float_format="%.3f"))

    # append the record to the screening report (replacing an earlier copy)
    md_path = os.path.join(PROC, "env9_screening.md")
    md = open(md_path, encoding="utf-8").read().split("\n## 9. Sensitivity")[0].rstrip()
    sec = ["", "", "## 9. Sensitivity - Rule 3a with water pixels excluded (step24b)", "",
           "Report only; the screening outcome above stands on the pre-specified, "
           "unmasked method shared with the existing six.", "",
           "| variable | r vs Fort, unmasked (screened) | r vs Fort, land only | "
           "r vs water_occurrence_pct, unmasked | land only |", "|---|---|---|---|---|"]
    for _, o in out.iterrows():
        sec.append(f"| `{o.variable}` | {o.r_fort_unmasked:+.3f} | {o.r_fort_land_only:+.3f} | "
                   f"{o.r_water_occ_unmasked:+.3f} | {o.r_water_occ_land_only:+.3f} |")
    flips = [o.variable for _, o in out.iterrows()
             if (abs(o.r_fort_unmasked) <= 0.8) != (abs(o.r_fort_land_only) <= 0.8)]
    sec += ["", ("**Rule 3a outcome changes under the land mask for: "
                 + ", ".join(f"`{v}`" for v in flips) + ".** The Rule 3a pass under the screened "
                 "method depends on sea and lagoon pixels in the coastal buffers lowering "
                 "the values there; on land alone the variable restates Distance from fort "
                 "above the 0.80 threshold, as `tree_cover_fraction` (r = +0.83) and "
                 "`built_up_fraction` (r = -0.81) did in the original screening.")
            if flips else "No outcome changes under the land mask.", ""]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md + "\n".join(sec))


if __name__ == "__main__":
    main()

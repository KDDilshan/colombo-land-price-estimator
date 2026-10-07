"""Step 1f - Diagnostic for R11 (de-duplication of re-posted adverts).

READ-ONLY.  Nothing is dropped here.  The question this answers is the one that
decides whether the R11 key is safe:

    of the groups that share Title + Address_Raw + Land_Size_Perch + Price_LKR
    + Land_Type + gn_division, how many are plausibly GENUINE SUBDIVIDED
    DEVELOPMENTS (several real plots advertised identically) rather than one
    advert re-posted?

Evidence used, pulled from the source file via `source_row_id`:

  Description  identical across the group  -> re-post, not separate plots
  Slug         a re-post normally reuses or bumps the same slug; genuinely
               separate plots usually carry distinct slugs
  Posted_Date  a re-post is spread over many dates; a real subdivision is
               usually listed on one or a few dates
  plot / lot markers in Title or Description that differ within the group
               would be positive evidence of separate plots

Output: results/step1f_dedup_diagnostic.txt

Run:  python src/step1f_dedup_diagnostic.py
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"

SOURCE = Path(r"C:\Users\HP\Desktop\Reserch\THESIS_FINAL\data\dataset_v6.csv")

KEY = [
    "Title", "Address_Raw", "Land_Size_Perch", "Price_LKR",
    "Land_Type", "gn_division",
]

# Markers that would distinguish one plot of a development from another.
PLOT_MARKER_RE = re.compile(
    r"\b(?:lot|plot|block|no\.?|number)\s*[:#]?\s*([0-9]{1,3}[a-z]?)\b", re.IGNORECASE
)


def main() -> int:
    # The pre-de-duplication snapshot written by step1e: R1-R3, R5 and R10
    # already applied, R11 not yet.  This is exactly the population R11 acts on.
    src = INTERIM / "listings_gn_predup.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run step1e_clean_listings.py first.")

    df = pd.read_csv(src, low_memory=False)

    # Pull Slug and Description from the source purely as evidence.
    extra = pd.read_csv(
        SOURCE, low_memory=False, usecols=["Slug", "Description", "Posted_Date"]
    )
    extra["source_row_id"] = extra.index
    df = df.merge(
        extra[["source_row_id", "Slug", "Description"]], on="source_row_id", how="left"
    )

    # Build the key as a single string column so the underlying numeric
    # columns keep their dtypes for reporting.
    df["_key"] = (
        df[KEY].astype(object).where(df[KEY].notna(), "<NA>").astype(str).agg("|".join, axis=1)
    )
    grp = df.groupby("_key", sort=False)

    sizes = grp.size()
    dup_keys = sizes[sizes > 1]

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(s)

    emit("=" * 78)
    emit("STEP 1f - R11 DE-DUPLICATION DIAGNOSTIC (read-only, nothing dropped)")
    emit("=" * 78)
    emit(f"rows after R10 (per-acre rows removed) : {len(df)}")
    emit(f"distinct key groups                    : {len(sizes)}")
    emit(f"groups with more than one row          : {len(dup_keys)}")
    emit(f"rows inside those groups               : {int(dup_keys.sum())}")
    emit(f"rows that R11 would remove             : {int(dup_keys.sum() - len(dup_keys))}")
    emit(f"rows remaining after R11               : {len(df) - int(dup_keys.sum() - len(dup_keys))}")
    emit("")
    emit(f"largest group                          : {int(sizes.max())} rows")
    emit("")
    emit("group size distribution:")
    for size, n in sizes[sizes > 1].value_counts().sort_index().items():
        emit(f"  {size:>4} rows x {n:>4} groups = {size * n:>5} rows")
    emit("")

    # ---------------------------------------------------------------- evidence
    emit("=" * 78)
    emit("EVIDENCE PER DUPLICATE GROUP")
    emit("=" * 78)

    rows = []
    for key, g in grp:
        if len(g) < 2:
            continue
        desc = g.Description.fillna("<NA>")
        slug = g.Slug.fillna("<NA>")
        dates = pd.to_datetime(g.Posted_Date, errors="coerce")
        markers = set()
        for txt in list(g.Title.fillna("")) + list(desc):
            markers |= {m.group(1).lower() for m in PLOT_MARKER_RE.finditer(str(txt))}
        span = (
            (dates.max() - dates.min()).days
            if dates.notna().sum() >= 2 else 0
        )
        texts = desc.tolist()
        sims = [
            difflib.SequenceMatcher(None, texts[0], t).ratio() for t in texts[1:]
        ]
        rows.append(
            {
                "min_desc_similarity": min(sims) if sims else 1.0,
                "mean_desc_similarity": sum(sims) / len(sims) if sims else 1.0,
                "n_rows": len(g),
                "n_desc": desc.nunique(),
                "n_slug": slug.nunique(),
                "n_dates": dates.dt.date.nunique(),
                "date_span_days": span,
                "n_plot_markers": len(markers),
                "gn_division": g.gn_division.iloc[0],
                "title": str(g.Title.iloc[0])[:70],
                "size": g.Land_Size_Perch.iloc[0],
                "price": g.Price_LKR.iloc[0],
            }
        )
    ev = pd.DataFrame(rows)

    identical_desc = ev[ev.n_desc == 1]
    differing_desc = ev[ev.n_desc > 1]
    identical_slug = ev[ev.n_slug == 1]

    emit(f"groups where every row has an IDENTICAL Description : "
         f"{len(identical_desc):>4}  ({len(identical_desc)/len(ev)*100:.1f}%)")
    emit(f"  rows they contain                                 : {int(identical_desc.n_rows.sum()):>4}")
    emit(f"groups where Description DIFFERS between rows       : "
         f"{len(differing_desc):>4}  ({len(differing_desc)/len(ev)*100:.1f}%)")
    emit(f"  rows they contain                                 : {int(differing_desc.n_rows.sum()):>4}")
    emit("")
    emit(f"groups where every row has an IDENTICAL Slug        : {len(identical_slug):>4}")
    emit(f"groups where Slug differs                           : {len(ev) - len(identical_slug):>4}")
    emit("")

    # Positive evidence of separate plots: differing plot/lot markers.
    genuine = ev[(ev.n_desc > 1) & (ev.n_plot_markers > 1)]
    emit("POSITIVE evidence of genuinely separate plots")
    emit("(Description differs AND more than one distinct lot/plot/block number")
    emit(" appears within the group):")
    emit(f"  groups : {len(genuine)}")
    emit(f"  rows   : {int(genuine.n_rows.sum()) if len(genuine) else 0}")
    emit(f"  rows R11 would wrongly remove : "
         f"{int(genuine.n_rows.sum() - len(genuine)) if len(genuine) else 0}")
    emit("")
    if len(genuine):
        emit("  these groups, largest first:")
        for r in genuine.sort_values("n_rows", ascending=False).head(40).itertuples():
            emit(f"    n={r.n_rows:>3} desc={r.n_desc:>3} slug={r.n_slug:>3} "
                 f"dates={r.n_dates:>3} markers={r.n_plot_markers:>2}  "
                 f"{r.gn_division:<18} {r.title}")
        emit("")

    emit("HOW DIFFERENT are the differing Descriptions?")
    emit("Sequence similarity of every row in a group against its first row.")
    emit("A re-post that only gained or lost a decorative line scores ~0.99.")
    emit("")
    for thr in (0.99, 0.95, 0.90, 0.80, 0.60):
        n = int((ev.min_desc_similarity >= thr).sum())
        emit(f"  groups whose rows are ALL >= {thr:.2f} similar : {n:>4} of {len(ev)}")
    emit("")
    hetero = ev[ev.mean_desc_similarity < 0.60]
    emit(f"groups whose MEAN within-group similarity is < 0.60 : {len(hetero)}")
    emit(f"  rows they contain              : {int(hetero.n_rows.sum())}")
    emit(f"  rows R11 removes from them     : {int(hetero.n_rows.sum() - len(hetero))}")
    emit(f"  share of all R11 removals      : "
         f"{(hetero.n_rows.sum() - len(hetero)) / max(int(dup_keys.sum() - len(dup_keys)), 1) * 100:.1f}%")
    emit("")
    if len(hetero):
        emit("  these groups in full (same seller, same development, rewritten copy):")
        emit(f"  {'rows':>5}{'dates':>7}{'minsim':>8}{'meansim':>9}  {'GN division':<18}"
             f"{'size':>7}{'price':>14}  title")
        for r in hetero.sort_values("n_rows", ascending=False).itertuples():
            emit(f"  {r.n_rows:>5}{r.n_dates:>7}{r.min_desc_similarity:>8.3f}"
                 f"{r.mean_desc_similarity:>9.3f}  {r.gn_division:<18}"
                 f"{r.size:>7.1f}{r.price:>14,.0f}  {r.title}")
        emit("")

    emit("Largest 25 duplicate groups:")
    emit("")
    emit(f"{'rows':>5}{'desc':>6}{'slug':>6}{'dates':>7}{'span_d':>8}{'marks':>7}  "
         f"{'GN division':<18}{'size':>7}{'price':>14}  title")
    emit("-" * 130)
    for r in ev.sort_values("n_rows", ascending=False).head(25).itertuples():
        emit(f"{r.n_rows:>5}{r.n_desc:>6}{r.n_slug:>6}{r.n_dates:>7}"
             f"{r.date_span_days:>8}{r.n_plot_markers:>7}  {r.gn_division:<18}"
             f"{r.size:>7.1f}{r.price:>14,.0f}  {r.title}")
    emit("")

    emit("Interpretation key:")
    emit("  desc=1  every row carries the same Description -> one advert, re-posted")
    emit("  slug=1  every row carries the same URL slug    -> one advert, re-posted")
    emit("  marks   distinct lot/plot/block numbers found  -> >1 means possible")
    emit("          separate plots; 0 or 1 means no such evidence")
    emit("")

    ev.sort_values("n_rows", ascending=False).to_csv(
        RESULTS / "step1f_dedup_groups.csv", index=False, encoding="utf-8"
    )
    (RESULTS / "step1f_dedup_diagnostic.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote results/step1f_dedup_diagnostic.txt")
    print(f"Wrote results/step1f_dedup_groups.csv ({len(ev)} duplicate groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

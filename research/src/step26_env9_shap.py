"""Step 26 - SHAP for the retrained model, and per-listing examples of how the
environmental and spatial variables contribute to one predicted price.

MODEL.  The `env_selected` configuration of step25 (existing six + passing new
variables; identical to the deployed model when none pass), fitted exactly as
step25.run_random fits it: random 80/20 split, seed 42, deploy/model.pkl
hyperparameters. SHAP is computed under the random split only - the same
caveat Chapter 6 states: it explains the model, not behaviour on unseen
divisions, and it describes associations the model uses, not causal effects.

SHAP values are in log(price per perch) units. A contribution phi multiplies
the predicted price by exp(phi); the examples report that as a percentage.

EXAMPLE SELECTION - by a feature rule fixed in advance, never by SHAP size:
  A  the test listing in the test-set division with the LOWEST hand_m
     (low-lying end); among its listings, the median predicted price
  B  the same for the division with the HIGHEST hand_m (upland end)
Each example's total environmental contribution is then placed as a
percentile of the whole test set, so a reader can see whether it is typical.

Outputs (results/env9/)
  shap_values.npy, shap_rows.csv       explained rows and their values
  shap_global.csv                      mean |SHAP| per feature, with block
  shap_blocks.csv                      share of mean |SHAP| per feature block
  shap_env_direction.csv               Spearman(feature value, SHAP) per env var
  fig_env9_shap_beeswarm.png           environmental variables only
  fig_env9_example_A.png / _B.png      grouped waterfall per listing
  shap_examples.md                     the two examples, in words and numbers
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step16_tuned_ablation_repeated as s16          # noqa: E402
import step25_env9_ablation as s25                    # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "env9")
CACHE = os.path.join(OUT, ".cache")
CHUNK = 50
SEED = 42
DPI = 300


def block_of(c, env):
    if c in env:
        return "environmental and spatial"
    if c == "Address_target_enc":
        return "division location (target encoding)"
    if c == "Distance_from_fort":
        return "distance from Fort"
    if c.startswith("Land_type_") or c == "Land_size_Perches_":
        return "plot size and land type"
    return "amenities and access"


def main():
    os.makedirs(CACHE, exist_ok=True)
    screen = json.load(open(os.path.join(PROC, "env9_screening.json")))
    env = s16.HAZARD_COLS + screen["passing_new"]
    X, y, groups, meta, prov = s25.build_matrix()
    assert prov["rows_used"] == 6167
    allc = list(X.columns)
    cols = [c for c in allc if c not in s25.EXISTING + s25.NEW or c in env]

    idx = np.arange(len(X))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=SEED)
    oof, enc = s16.target_encode_oof(groups.iloc[itr], y.iloc[itr], groups.iloc[ite], SEED)
    Xtr, Xte = X.iloc[itr][cols].copy(), X.iloc[ite][cols].copy()
    Xtr["Address_target_enc"] = oof.values
    Xte["Address_target_enc"] = enc.values
    m = RandomForestRegressor(random_state=SEED, **s16.TUNED).fit(Xtr, y.iloc[itr])
    pred = m.predict(Xte)
    r2 = 1 - ((y.iloc[ite] - pred) ** 2).sum() / ((y.iloc[ite] - y.iloc[ite].mean()) ** 2).sum()
    print(f"env block {env}\nheld-out R2 {r2 * 100:.3f}%  ({len(cols) + 1} features)", flush=True)

    # --------------------------------------------------- SHAP, chunked + cached
    feats = list(Xte.columns)
    tag = f"{'_'.join(screen['passing_new']) or 'env6'}_{len(feats)}"
    path = os.path.join(CACHE, f"shap_{tag}.npy")
    sv = np.load(path) if os.path.exists(path) else np.full(Xte.shape, np.nan)
    ex = shap.TreeExplainer(m)
    base = float(np.ravel(ex.expected_value)[0])
    t0 = time.time()
    for s in range(0, len(Xte), CHUNK):
        if not np.isnan(sv[s]).any():
            continue
        sv[s:s + CHUNK] = ex.shap_values(Xte.iloc[s:s + CHUNK], check_additivity=False)
        np.save(path, sv)
        print(f"  shap rows {min(s + CHUNK, len(Xte))}/{len(Xte)}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    add_err = float(np.abs(base + sv.sum(1) - pred).max())
    print(f"additivity max |base + sum(phi) - prediction| = {add_err:.2e}")

    np.save(os.path.join(OUT, "shap_values.npy"), sv)
    rows = meta.iloc[ite].reset_index(drop=True).assign(
        test_row=np.arange(len(ite)), log_pred=pred, pred_rs=np.exp(pred),
        env_contrib=sv[:, [feats.index(c) for c in env]].sum(1))
    rows.to_csv(os.path.join(OUT, "shap_rows.csv"), index=False)

    # ------------------------------------------------------------ global view
    g = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)})
    g["block"] = [block_of(c, env) for c in feats]
    g["share_pct"] = 100 * g.mean_abs_shap / g.mean_abs_shap.sum()
    g["rank"] = g.mean_abs_shap.rank(ascending=False).astype(int)
    g = g.sort_values("mean_abs_shap", ascending=False)
    g.to_csv(os.path.join(OUT, "shap_global.csv"), index=False, float_format="%.6f")
    blocks = (g.groupby("block").agg(mean_abs_shap=("mean_abs_shap", "sum"),
                                    share_pct=("share_pct", "sum"),
                                    n_features=("feature", "count"))
              .sort_values("share_pct", ascending=False))
    blocks.to_csv(os.path.join(OUT, "shap_blocks.csv"), float_format="%.4f")

    direc = []
    for c in env:
        j = feats.index(c)
        rho, p = spearmanr(Xte[c], sv[:, j])
        direc.append(dict(variable=c, rank=int(g.set_index("feature").loc[c, "rank"]),
                          mean_abs_shap=float(np.abs(sv[:, j]).mean()),
                          spearman_value_vs_shap=float(rho), p=float(p)))
    direc = pd.DataFrame(direc)
    direc.to_csv(os.path.join(OUT, "shap_env_direction.csv"), index=False, float_format="%.5f")
    print(blocks.to_string(float_format="%.3f"))
    print(direc.to_string(index=False, float_format="%.4f"))

    jj = [feats.index(c) for c in env]
    shap.summary_plot(sv[:, jj], Xte[env], feature_names=env, show=False, plot_size=(8, 4.5))
    plt.title("SHAP values - environmental and spatial variables\n"
              "(log price per perch; held-out listings, random split)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_env9_shap_beeswarm.png"), dpi=DPI)
    plt.close()

    # ------------------------------------------------------------- examples
    div_hand = Xte.assign(Address=rows.Address.values).groupby("Address")["hand_m"].first()
    pct = lambda v: float((np.abs(rows.env_contrib) <= abs(v)).mean() * 100)
    md = ["# Per-listing SHAP examples - environmental and spatial variables", "",
          f"Model: `env_selected` ({len(feats)} features, env block "
          f"{', '.join(f'`{c}`' for c in env)}), random 80/20 split seed 42, "
          f"held-out R² {r2 * 100:.2f}%. Base value (mean prediction) "
          f"Rs {np.exp(base):,.0f} per perch.", "",
          "Listings chosen by a rule fixed in advance: the median-priced test listing "
          "in the test division with the lowest `hand_m` (A) and the highest (B). "
          "Contributions are SHAP values in log price; exp(φ) − 1 is the percentage "
          "change they make to the predicted price. They describe what the model "
          "associates with the price, not a causal effect.", ""]
    for lab, addr in (("A", div_hand.idxmin()), ("B", div_hand.idxmax())):
        sub = rows[rows.Address == addr].sort_values("log_pred")
        r = sub.iloc[(len(sub) - 1) // 2]
        i = int(r.test_row)
        phi = pd.Series(sv[i], index=feats)
        grp = phi.groupby([block_of(c, env) for c in feats]).sum()
        steps = [(k, grp[k]) for k in ("division location (target encoding)",
                                       "distance from Fort", "amenities and access",
                                       "plot size and land type")]
        steps += [(c, phi[c]) for c in env]
        waterfall(lab, addr, base, steps, float(r.log_pred), float(r["Price per Perch"]),
                  Xte.iloc[i], env)
        e_tot = float(phi[env].sum())
        md += [f"## Example {lab} - {addr.title()} (test row {i})", "",
               f"Asking price Rs {r['Price per Perch']:,.0f} per perch; predicted "
               f"Rs {r.pred_rs:,.0f}; plot {Xte.iloc[i]['Land_size_Perches_']:g} perches.", "",
               "| component | value | SHAP (log) | effect on price |", "|---|---|---|---|",
               f"| base value | | | Rs {np.exp(base):,.0f} |"]
        for k, v in steps:
            val = f"{Xte.iloc[i][k]:.3f}" if k in env else ""
            md.append(f"| {k} | {val} | {v:+.4f} | {np.expm1(v) * 100:+.2f}% |")
        md += [f"| **environmental block, total** | | **{e_tot:+.4f}** | "
               f"**{np.expm1(e_tot) * 100:+.2f}%** (≈ Rs {r.pred_rs - r.pred_rs / np.exp(e_tot):+,.0f} per perch) |",
               f"| prediction | | {r.log_pred - base:+.4f} | Rs {r.pred_rs:,.0f} |", "",
               f"The environmental block's absolute contribution here is larger than in "
               f"{pct(e_tot):.0f}% of held-out listings.", ""]
    md += ["## Test-set context", "",
           f"Across all {len(rows)} held-out listings the environmental block's total "
           f"contribution has median |φ| {np.median(np.abs(rows.env_contrib)):.4f} "
           f"(≈ {np.expm1(np.median(np.abs(rows.env_contrib))) * 100:.2f}% of price) and "
           f"95th percentile {np.percentile(np.abs(rows.env_contrib), 95):.4f} "
           f"(≈ {np.expm1(np.percentile(np.abs(rows.env_contrib), 95)) * 100:.2f}%). "
           f"The block holds {blocks.loc['environmental and spatial', 'share_pct']:.2f}% of "
           f"total mean |SHAP|; division location holds "
           f"{blocks.loc['division location (target encoding)', 'share_pct']:.2f}%.", ""]
    with open(os.path.join(OUT, "shap_examples.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    json.dump(dict(r2_pct=r2 * 100, base_log=base, base_rs=float(np.exp(base)),
                   additivity_max_err=add_err, env=env, n_features=len(feats),
                   n_explained=int(len(Xte))),
              open(os.path.join(OUT, "shap_meta.json"), "w"), indent=2)
    print(f"wrote {OUT}")


def waterfall(lab, addr, base, steps, final, actual, xrow, env):
    """Horizontal waterfall on the log scale, labelled as % change in price."""
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * (len(steps) + 2) + 1.2))
    cum = base
    names = ["base value"]
    lefts, widths, colors = [0], [0], ["#888888"]
    for k, v in steps:
        lefts.append(cum if v >= 0 else cum + v)
        widths.append(abs(v))
        colors.append("#1a7f5a" if v >= 0 else "#b03a2e")
        names.append(f"{k} = {xrow[k]:.3g}" if k in env else k)
        cum += v
    names.append("prediction")
    lefts.append(0)
    widths.append(0)
    colors.append("#888888")
    ypos = np.arange(len(names))[::-1]
    for yy, l, w, c in zip(ypos, lefts, widths, colors):
        ax.barh(yy, w, left=l, color=c, height=0.6)
    for n, (k, v) in enumerate(steps, start=1):
        ax.text(lefts[n] + widths[n] + 0.005, ypos[n],
                f"{np.expm1(v) * 100:+.1f}%", va="center", fontsize=7.5)
    ax.axvline(base, color="#888888", lw=0.8, ls=":")
    ax.axvline(final, color="#333333", lw=0.8, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    for t, nm in zip(ax.get_yticklabels(), names):
        if any(nm.startswith(e) for e in env):
            t.set_color("#1f4e99")
    ax.set_xlabel("log(price per perch)")
    lo = min(lefts[1:-1] + [base, final]) - 0.1
    hi = max([l + w for l, w in zip(lefts[1:-1], widths[1:-1])] + [base, final]) + 0.25
    ax.set_xlim(lo, hi)
    ax.set_title(f"Example {lab}: {addr.title()} - base Rs {np.exp(base):,.0f} -> predicted "
                 f"Rs {np.exp(final):,.0f} per perch (asking Rs {actual:,.0f})\n"
                 "environmental and spatial variables in blue; associations, not causal effects",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_env9_example_{lab}.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()

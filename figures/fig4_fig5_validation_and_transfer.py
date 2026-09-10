#!/usr/bin/env python
"""
fig4_fig5_validation_and_transfer.py

Produces Figures 4 and 5 of the manuscript: one script, because both figures
read the same response-property and synthesis products and share their plotting helpers.

  --out-validation  Observation-side validation of the model-derived filter properties.
      (a) memory: e-folding tau of observed vs simulated flow, log-log, 1:1 line.
      (b) timing: late-response fraction of observed vs simulated flow.
      (c) the observed late-response fraction against catchment snow cover, with
          binned medians. Panels (a) and (b) answer "is this HBV or is this the
          catchment"; panel (c) shows the snowmelt phase shift in the data alone.

  --out-transfer    The physiographic synthesis and what actually transfers.
      (a) standardised coefficients for the three filter properties fitted on
          OBSERVED flow, with country-clustered 95 % intervals; filled markers survive
          Benjamini-Hochberg control at q<0.05 on the restricted wild cluster bootstrap
          p-values at G=16 (family = the six predictors within a response).
      (b) leave-one-country-out skill for observed memory, per country, scored three
          ways: against the global mean (the flattering baseline), against each held-out
          country's own mean (the honest one), and that same local score with the
          national level offset removed, which is the part attributable to failing to
          ORDER the catchments rather than to misplacing the country.
      (c) the same contrast summarised for all three properties.

  Panel (b)/(c) point: a transferred relationship fails two ways at once, and the two
  have opposite remedies -- a few local gauges fix a level error, nothing national fixes
  an ordering error. The gap between the red and orange bars is the level component.

    python figures/fig4_fig5_validation_and_transfer.py --strength-memory <response_strength_memory_DJF.parquet> \
        --timing <response_timing_DJF.parquet> --observed <response_observed_DJF.parquet> \
        --summary <physiographic_summary_full.csv> --coeffs <physiographic_coeffs_full.csv> \
        --by-country <physiographic_by_country_full.csv> \
        [--out-validation <fig4.png>] [--out-transfer <fig5.png>]

  With neither given, writes the published Figure 4 and Figure 5 paths (and the
  .pdfs beside them). Run from the repository root: those defaults are relative
  to it.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# AGU asks for >=8 pt text at the printed size and will not take Type 3 fonts, so
# both figures are drawn at the full-page width (6.5 in / 39 pc) and saved without
# a tight bounding box: what is set here is what reaches the page, unscaled.
FIG_W = 6.5
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, not Type 3
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans",
                        "DejaVu Sans"],
    # 9 pt base, not 8: mathtext draws exponents at 0.7x, so an 8 pt "$R^2$"
    # puts the 2 at 5.6 pt, under AGU's 6 pt floor for super/subscripts. At 9 pt
    # the smallest mark on the page is 6.3 pt.
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    # mathtext defaults to DejaVu whatever font.family says, and stixsans draws
    # the tau from STIXNonUnicode, where it has no code point and will not
    # extract. "custom" keeps the maths in the text face, which has the Greek.
    "mathtext.fontset": "custom", "mathtext.default": "it",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans", "mathtext.tt": "monospace", "mathtext.sf": "sans",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
})

PRED_LABEL = {"slope_l": "slope (log)", "frac_snow": "snow fraction",
              "gwt_l": "GW-table depth (log)", "aridity": "aridity",
              "karst_pc": "karst %", "clay_pc": "clay %"}
PROPS = [("gain_Qobs", "response strength", "#0072b2", "o"),
         ("log_tau_obs", "memory", "#009e73", "s"),
         ("reg_lag_obs", "timing", "#d55e00", "^")]
GLOBAL_C, LOCAL_C, DEBIAS_C = "#b9bcc0", "#c44e52", "#e8a33d"
MIN_COUNTRY_N = 10   # per-country panel: skip groups too small for a meaningful R2


def _save(fig, path):
    # no bbox_inches="tight": it crops the canvas away from figsize, and the point
    # of drawing at the final width is that the width is known
    fig.savefig(path, dpi=400)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
    plt.close(fig)
    print(f"wrote {path} ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)", flush=True)


# ------------------------------------------------------------------- validation

def _stat(ax, text):
    """Spearman/n line, in-panel rather than in a second title line."""
    ax.text(0.04, 0.965, text, transform=ax.transAxes, va="top",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def fig_validation(d, path):
    # three near-square panels across the full page width
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.62))
    fig.subplots_adjust(left=0.085, right=0.980, bottom=0.175, top=0.90, wspace=0.44)
    SC = dict(s=2.0, alpha=0.28, lw=0, color="#2f4b7c")   # denser panels, smaller dots
    MED = dict(color="#c44e52", lw=1.2, ms=2.6)

    ax = axes[0]
    m = d[["tau_Qobs", "tau_Qsim_m"]].dropna()
    ax.scatter(m.tau_Qobs, m.tau_Qsim_m, **SC)
    lim = [max(1.0, m.min().min() * 0.8), m.max().max() * 1.2]
    ax.plot(lim, lim, "k--", lw=0.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=r"observed flow memory $\tau$ (days)",
           ylabel=r"simulated flow memory $\tau$ (days)")
    rho = stats.spearmanr(m.tau_Qobs, m.tau_Qsim_m).statistic
    bias = np.median(m.tau_Qsim_m / m.tau_Qobs)
    ax.set_title("(a) memory", loc="left")
    _stat(ax, f"Spearman {rho:+.2f}, n={len(m)}\nmodel {bias:.2f}$\\times$ observed")

    ax = axes[1]
    m = d[["late_frac_obs", "late_frac"]].dropna()
    ax.scatter(m.late_frac_obs, m.late_frac, **SC)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.25, 0.5, 0.75, 1.0],
           yticks=[0, 0.25, 0.5, 0.75, 1.0],
           xlabel="observed late-response fraction",
           ylabel="simulated late-response fraction")
    # equal-count binned medians, as in (c): the cloud is broad and the eye needs
    # a guide to see the +0.71 rank agreement in it
    xb = m.late_frac_obs.to_numpy(); yb = m.late_frac.to_numpy()
    edges = np.unique(np.nanquantile(xb, np.linspace(0, 1, 10)))
    cx, cy = [], []
    for i in range(len(edges) - 1):
        sel = (xb >= edges[i]) & (xb < edges[i + 1] if i < len(edges) - 2 else xb <= edges[i + 1])
        if sel.sum() >= 20:
            cx.append(np.nanmedian(xb[sel])); cy.append(np.nanmedian(yb[sel]))
    ax.plot(cx, cy, "-o", zorder=4, **MED)
    rho = stats.spearmanr(m.late_frac_obs, m.late_frac).statistic
    ax.set_title("(b) timing", loc="left")
    _stat(ax, f"Spearman {rho:+.2f}, n={len(m)}")

    ax = axes[2]
    m = d[["late_frac_obs", "sp_active_frac"]].dropna()
    ax.scatter(m.sp_active_frac, m.late_frac_obs, **SC)
    # same equal-count scheme as (b), with the snow-free atom held out: equal-width
    # bins left the upper ones nearly empty and the median line wandered
    sx = m.sp_active_frac.to_numpy(); sy = m.late_frac_obs.to_numpy()
    xs, ys = [], []
    zero = sx <= 0
    if zero.sum() >= 20:
        xs.append(0.0); ys.append(np.nanmedian(sy[zero]))
    pos = sx > 0
    if pos.sum() >= 40:
        ed = np.unique(np.nanquantile(sx[pos], np.linspace(0, 1, 9)))
        for i in range(len(ed) - 1):
            last = i == len(ed) - 2
            sel = pos & (sx >= ed[i]) & ((sx <= ed[i + 1]) if last else (sx < ed[i + 1]))
            if sel.sum() >= 20:
                xs.append(np.nanmedian(sx[sel])); ys.append(np.nanmedian(sy[sel]))
    ax.plot(xs, ys, "-o", zorder=4, **MED)
    rho = stats.spearmanr(m.sp_active_frac, m.late_frac_obs).statistic
    ax.set(xlabel="snow-active fraction of winters",
           ylabel="observed late-response fraction",
           ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0],
           xlim=(-0.02, 1.02), xticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(c) late fraction vs snow", loc="left")
    _stat(ax, f"Spearman {rho:+.2f}")

    for a in axes:
        a.grid(alpha=0.25, lw=0.4)
    _save(fig, path)


# --------------------------------------------------------------------- transfer

def fig_transfer(summary, coeffs, by_country, path):
    # 2 rows at full-page width: (a) and (c) share the top, (b) takes the whole
    # bottom -- 14 country names will not fit in a third of 6.5 in at 8 pt.
    fig = plt.figure(figsize=(FIG_W, 5.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.34], width_ratios=[1.12, 1.0])
    fig.subplots_adjust(left=0.208, right=0.985, top=0.955, bottom=0.150,
                        hspace=0.55, wspace=0.42)

    # (a) coefficients for the three observed-flow responses
    ax = fig.add_subplot(gs[0, 0])
    order = ["slope_l", "gwt_l", "aridity", "karst_pc", "frac_snow", "clay_pc"]
    h = 0.26
    for k, (resp, lab, col, mk) in enumerate(PROPS):
        c = coeffs[coeffs.response == resp].set_index("predictor")
        if c.empty:
            continue
        y = np.arange(len(order)) + (k - 1) * h
        b = c.loc[order, "beta_std"].to_numpy()
        e = 1.96 * c.loc[order, "se_clustered"].to_numpy()
        # significance is FDR-controlled across the six predictors of this response
        sigcol = "q_bh" if "q_bh" in c.columns else "p_wild_bootstrap"
        sig = c.loc[order, sigcol].to_numpy() < 0.05
        ax.errorbar(b, y, xerr=e, fmt="none", ecolor=col, elinewidth=1.0, alpha=0.9)
        ax.scatter(b[sig], y[sig], s=16, color=col, marker=mk, zorder=4, label=lab)
        ax.scatter(b[~sig], y[~sig], s=16, facecolors="white", edgecolors=col,
                   marker=mk, linewidths=0.9, zorder=4)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([PRED_LABEL[p] for p in order])
    ax.invert_yaxis()
    ax.set_ylim(len(order) + 0.05, -0.55)   # blank strip at the foot for the note
    ax.set_xlabel("standardised coefficient (observed flow)")
    ax.set_title("(a) physiographic controls", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.text(0.015, 0.025, "filled: $q<0.05$ (BH)", transform=ax.transAxes,
            style="italic", color="0.35")

    # (c) the summary contrast for all three properties
    ax = fig.add_subplot(gs[0, 1])
    labs, kf, lg, ll, ld = [], [], [], [], []
    has_db = "loco_r2_local_debiased_pooled" in summary.columns
    for resp, lab, _, _ in PROPS:
        s = summary[summary.response == resp]
        if s.empty:
            continue
        labs.append(lab)
        kf.append(float(s.kfold_r2_global.iloc[0]))
        lg.append(float(s.loco_r2_global.iloc[0]))
        ll.append(float(s.loco_r2_local_pooled.iloc[0]))
        ld.append(float(s.loco_r2_local_debiased_pooled.iloc[0]) if has_db else np.nan)
    x = np.arange(len(labs))
    w = 0.21
    series = [(kf, -1.5 * w, "#4c72b0", "10-fold (interpolation; (c) only)"),
              (lg, -0.5 * w, GLOBAL_C, "LOCO vs global mean"),
              (ll, 0.5 * w, LOCAL_C, "LOCO vs own-country mean")]
    if has_db:
        series.append((ld, 1.5 * w, DEBIAS_C, "LOCO vs own-country mean, de-biased"))
    handles = []
    for vals, off, colour, lab in series:
        handles.append(ax.bar(x + off, vals, w, color=colour, label=lab))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labs])
    # limits follow the data: nothing is clipped, so nothing needs a label
    # written up the inside of a bar at 6.5 pt
    lo = float(np.nanmin(np.concatenate([np.asarray(v, float) for v, *_ in series])))
    hi = float(np.nanmax(np.concatenate([np.asarray(v, float) for v, *_ in series])))
    pad = 0.07 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_title("(c) what transfers", loc="left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)

    # (b) per-country transfer for observed memory, both baselines
    ax = fig.add_subplot(gs[1, :])
    g = by_country[(by_country.response == "log_tau_obs") & (by_country.n >= MIN_COUNTRY_N)]
    g = g.sort_values("r2_local")
    y = np.arange(len(g))
    bars = [(-0.27, "r2_global", GLOBAL_C), (0.00, "r2_local", LOCAL_C),
            (0.27, "r2_local_debiased", DEBIAS_C)]
    vals = []
    for off, col, colour in bars:
        if col not in g.columns:
            continue
        ax.barh(y + off, g[col], height=0.25, color=colour)
        vals.append(np.asarray(g[col], float))
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(g.country)
    ax.set_ylim(-0.65, len(g) - 0.35)
    lo = float(np.nanmin(np.concatenate(vals))); hi = float(np.nanmax(np.concatenate(vals)))
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("leave-one-country-out $R^2$")
    ax.set_title("(b) observed memory, by withheld country", loc="left")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.text(0.012, 0.982, f"countries with n $\\geq$ {MIN_COUNTRY_N}",
            transform=ax.transAxes, style="italic", color="0.35", va="top")

    # one legend for the scoring baselines, which (b) and (c) share
    fig.legend(handles=[handles[1], handles[2], handles[3], handles[0]][:len(handles)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.004),
               columnspacing=1.6, handletextpad=0.5, handlelength=1.5)

    _save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    for f in ["strength-memory", "timing", "observed", "summary", "coeffs", "by-country"]:
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--out-validation",
                    default="figures/fig4_observed_validation.png")
    ap.add_argument("--out-transfer",
                    default="figures/fig5_physiography_and_transfer.png")
    a = ap.parse_args()

    d = pd.read_parquet(a.observed).merge(
        pd.read_parquet(a.strength_memory)[["gauge_id", "tau_Qsim", "ac1_Qsim", "sp_active_frac"]],
        on="gauge_id", how="left").merge(
        pd.read_parquet(a.timing)[["gauge_id", "late_frac", "reg_lag", "sig"]],
        on="gauge_id", how="left")
    d = d[(d.sig > 0.2) | d.sig.isna()]
    fig_validation(d, a.out_validation)
    fig_transfer(pd.read_csv(a.summary), pd.read_csv(a.coeffs),
                 pd.read_csv(a.by_country), a.out_transfer)


if __name__ == "__main__":
    main()

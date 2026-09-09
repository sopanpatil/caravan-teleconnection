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

PRED_LABEL = {"slope_l": "slope (log)", "frac_snow": "snow fraction",
              "gwt_l": "GW-table depth (log)", "aridity": "aridity",
              "karst_pc": "karst %", "clay_pc": "clay %"}
PROPS = [("gain_Qobs", "response strength", "#0072b2", "o"),
         ("log_tau_obs", "memory", "#009e73", "s"),
         ("reg_lag_obs", "timing", "#d55e00", "^")]
GLOBAL_C, LOCAL_C, DEBIAS_C = "#b9bcc0", "#c44e52", "#e8a33d"
MIN_COUNTRY_N = 10   # per-country panel: skip groups too small for a meaningful R2


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


# ------------------------------------------------------------------- validation

def fig_validation(d, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    m = d[["tau_Qobs", "tau_Qsim_m"]].dropna()
    ax.scatter(m.tau_Qobs, m.tau_Qsim_m, s=6, alpha=0.25, lw=0, color="#2f4b7c")
    lim = [max(1.0, m.min().min() * 0.8), m.max().max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=r"observed flow memory $\tau$ (days)",
           ylabel=r"simulated flow memory $\tau$ (days)")
    rho = stats.spearmanr(m.tau_Qobs, m.tau_Qsim_m).statistic
    bias = np.median(m.tau_Qsim_m / m.tau_Qobs)
    ax.set_title(f"(a) memory\nSpearman {rho:+.2f}, n={len(m)}", fontsize=10, loc="left")
    ax.text(0.04, 0.93, f"model {bias:.2f}$\\times$ observed (median)",
            transform=ax.transAxes, fontsize=8.5, va="top")

    ax = axes[1]
    m = d[["late_frac_obs", "late_frac"]].dropna()
    ax.scatter(m.late_frac_obs, m.late_frac, s=6, alpha=0.25, lw=0, color="#2f4b7c")
    ax.plot([0, 1], [0, 1], "k--", lw=1, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1),
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
    ax.plot(cx, cy, "-o", color="#c44e52", lw=2, ms=4, zorder=4)
    rho = stats.spearmanr(m.late_frac_obs, m.late_frac).statistic
    ax.set_title(f"(b) timing\nSpearman {rho:+.2f}, n={len(m)}", fontsize=10, loc="left")

    ax = axes[2]
    m = d[["late_frac_obs", "sp_active_frac"]].dropna()
    ax.scatter(m.sp_active_frac, m.late_frac_obs, s=6, alpha=0.25, lw=0, color="#2f4b7c")
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
    ax.plot(xs, ys, "-o", color="#c44e52", lw=2, ms=4, zorder=4)
    rho = stats.spearmanr(m.sp_active_frac, m.late_frac_obs).statistic
    ax.set(xlabel="snow-active fraction of winters", ylabel="observed late-response fraction",
           ylim=(0, 1))
    ax.set_title(f"(c) observed late fraction vs snow-active winters\nSpearman {rho:+.2f}",
                 fontsize=10, loc="left")

    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------- transfer

def fig_transfer(summary, coeffs, by_country, path):
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.42)

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
        ax.errorbar(b, y, xerr=e, fmt="none", ecolor=col, elinewidth=1.4, alpha=0.9)
        ax.scatter(b[sig], y[sig], s=36, color=col, marker=mk, zorder=4, label=lab)
        ax.scatter(b[~sig], y[~sig], s=36, facecolors="white", edgecolors=col,
                   marker=mk, linewidths=1.3, zorder=4)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([PRED_LABEL[p] for p in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("standardised coefficient (observed flow)")
    ax.text(0.02, 0.02, "filled: $q<0.05$ (BH)", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="0.35")
    ax.set_title("(a) physiographic controls", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    # (b) per-country transfer for observed memory, both baselines
    ax = fig.add_subplot(gs[0, 1])
    g = by_country[(by_country.response == "log_tau_obs") & (by_country.n >= MIN_COUNTRY_N)]
    g = g.sort_values("r2_local")
    y = np.arange(len(g))
    bars = [(-0.27, "r2_global", GLOBAL_C, "vs global mean"),
            (0.00, "r2_local", LOCAL_C, "vs own-country mean"),
            (0.27, "r2_local_debiased", DEBIAS_C, "vs own-country mean, de-biased")]
    for off, col, colour, lab in bars:
        if col not in g.columns:
            continue
        ax.barh(y + off, np.clip(g[col], -1, 1), height=0.25, color=colour, label=lab)
    for i, (_, r) in enumerate(g.iterrows()):
        for off, col, _, _ in bars:
            v = r.get(col, np.nan)
            if np.isfinite(v) and v < -1:
                ax.text(-0.98, i + off, f"{v:.1f}", va="center", ha="left",
                        fontsize=6.5, color="white")
    ax.text(0.02, 0.02, f"countries with n $\\geq$ {MIN_COUNTRY_N}",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="0.35")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(g.country, fontsize=8)
    ax.set_xlim(-1.02, 1.0)
    ax.set_xlabel("leave-one-country-out $R^2$")
    ax.set_title("(b) observed memory, by withheld country", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1,
              frameon=False, handletextpad=0.4, columnspacing=1.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    # (c) the summary contrast for all three properties
    ax = fig.add_subplot(gs[0, 2])
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
    ax.bar(x - 1.5 * w, kf, w, color="#4c72b0", label="10-fold (interpolation)")
    ax.bar(x - 0.5 * w, lg, w, color=GLOBAL_C, label="LOCO vs global mean")
    ax.bar(x + 0.5 * w, np.clip(ll, -1, None), w, color=LOCAL_C,
           label="LOCO vs own-country mean")
    if has_db:
        ax.bar(x + 1.5 * w, np.clip(ld, -1, None), w, color=DEBIAS_C,
               label="LOCO vs own-country mean, de-biased")
    # bars are narrow, so a clipped value is written up the inside of its own bar
    for series, off in [(ll, 0.5 * w), (ld, 1.5 * w)]:
        for i, v in enumerate(series):
            if np.isfinite(v) and v < -1:
                ax.text(i + off, -0.985, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color="white", fontweight="bold", rotation=90)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labs])
    ax.set_ylim(-1.02, 0.55)
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_title("(c) what transfers", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              frameon=False, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

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

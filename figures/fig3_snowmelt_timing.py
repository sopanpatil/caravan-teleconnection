#!/usr/bin/env python
"""
fig3_snowmelt_timing.py

Produces Figure 3 of the manuscript.

Timing figure: the snowmelt phase shift. The winter teleconnection signal
is tracked across the whole water year, and the flow response separates three
registrations in time (fast winter / sustained aquifer / delayed snowmelt).

Three panels:
  (a) monthly response profiles r(L) for endpoint catchments (a flashy low-snow
      upland, a chalk aquifer, a Nordic snow catchment) -- the winter peak and the
      distinct spring-melt peak.
  (b) map of the late-response fraction (share of response energy in April+),
      the bounded phase-shift index -- snow-concentrated.
  (c) late-response fraction vs snow cover across the sample (the Spearman control).

    python figures/fig3_snowmelt_timing.py --timing <response_timing_DJF.parquet> \
        --attrs <attributes.parquet> --signal <precipitation_signal_DJF.parquet> \
        --indices <teleconnection_seasonal.csv> --states-dir <dir> \
        --manifest <states_manifest.csv> --out figures/fig3_snowmelt_timing.png
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# AGU asks for >=8 pt text at the printed size and will not take Type 3 fonts.
# Both are settled here: the figure is drawn at the full-page width (6.5 in /
# 39 pc) with a 9 pt base, and a tight bounding box can only ever crop the
# canvas smaller, so the printed text is 9 pt or a shade larger, never less.
FIG_W = 6.5
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType, not Type 3
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans",
                        "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    # mathtext defaults to DejaVu whatever font.family says; "custom" keeps the
    # tau and the subscripts in the text face, which carries the Greek
    "mathtext.fontset": "custom", "mathtext.default": "it",
    "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "mathtext.cal": "sans", "mathtext.tt": "monospace", "mathtext.sf": "sans",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

def _pdf_width_in(path):
    """Width of the saved PDF, in inches. A tight bounding box crops the canvas,
    so this reports what actually landed rather than what figsize asked for."""
    import re
    with open(path, "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[([^\]]*)\]", fh.read())
    return float(m.group(1).split()[2]) / 72 if m else float("nan")


EXTENT = [-25, 32, 34, 72]
BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
        "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}
LAGS = list(range(0, 12))
MLAB = ["Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
# endpoint catchments (fall back to nearest in-sample if absent)
ENDPOINTS = [("camelsgb", "camelsgb_15025", "flashy upland (Scotland, snow-free)", "#d55e00"),
             ("camelsgb", "camelsgb_43008", "chalk aquifer (England)", "#cc79a7"),
             ("grdc", "GRDC_6729140", "snow catchment (Norway)", "#0072b2")]


def _basemap(ax):
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f1ec", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dce7ef", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#555", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.25, edgecolor="#999", zorder=1)
    except Exception as e:  # noqa: BLE001
        print(f"  (cartopy unavailable: {type(e).__name__}; plain frame)", flush=True)


def _map_ax(fig, gs):
    try:
        import cartopy.crs as ccrs
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(gs, projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        return ax, {"transform": proj}
    except Exception:
        ax = fig.add_subplot(gs)
        ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1.4)
        return ax, {}


def lag_profile(gid, src, betas, indices, states_dir, spinup):
    """r(L): correlate winter forcing Phat'(y) with monthly flow anomaly at lag L."""
    b = betas.loc[gid]
    phat = sum(b[BETA[c]] * indices[c] for c in BETA).dropna()
    df = pd.read_parquet(os.path.join(states_dir, src, gid + ".parquet"), columns=["Qsim"])
    if gid in spinup.index:
        df = df[df.index >= spinup.loc[gid]]
    m = df["Qsim"].resample("MS").mean()
    m = m - m.groupby(m.index.month).transform("mean")
    idx = {(t.year, t.month): v for t, v in m.items()}
    r = []
    for L in LAGS:
        xs, ys = [], []
        for y, p in phat.items():
            if not np.isfinite(p):
                continue
            dt = pd.Timestamp(int(y) - 1, 12, 1) + pd.DateOffset(months=L)
            v = idx.get((dt.year, dt.month), np.nan)
            if np.isfinite(v):
                xs.append(v); ys.append(p)
        r.append(np.corrcoef(xs, ys)[0, 1] if len(xs) > 15 else np.nan)
    return np.array(r)


def make(s2c, attrs, betas, indices, states_dir, spinup, out):
    d = s2c.merge(attrs, on="gauge_id", how="left")
    ok = d[d.sig > 0.2].copy()

    fig = plt.figure(figsize=(FIG_W, FIG_W * 8.4 / 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.4], width_ratios=[1.25, 1.0],
                          hspace=0.26, wspace=0.36)

    # (a) response profiles for endpoints -------------------------------------
    axp = fig.add_subplot(gs[0, :])
    for src, gid, lab, col in ENDPOINTS:
        if gid not in betas.index:
            print(f"  endpoint {gid} not in sample; skipped", flush=True)
            continue
        try:
            r = lag_profile(gid, src, betas, indices, states_dir, spinup)
        except FileNotFoundError:
            print(f"  states missing for {gid}; skipped", flush=True)
            continue
        axp.plot(LAGS, r, "-o", color=col, lw=1.8, ms=4, label=lab)
    axp.axhline(0, color="#888", lw=0.7)
    axp.axvspan(-0.5, 2.5, color="#dce7ef", alpha=0.6, zorder=0)   # DJF window
    axp.text(1, axp.get_ylim()[1] * 0.92 if axp.get_ylim()[1] > 0 else 0.6, "winter",
             ha="center", color="#345")
    axp.set_xticks(LAGS); axp.set_xticklabels(MLAB)
    axp.set_ylabel("corr(flow anomaly, winter forcing)")
    axp.set_title("(a) Monthly response profiles $r(L)$ for three endpoint catchments",
                  loc="left")
    axp.legend(loc="upper right", framealpha=0.9)
    axp.grid(axis="y", ls=":", alpha=0.4)

    # (b) map of late-response fraction ---------------------------------------
    axm, tf = _map_ax(fig, gs[1, 0])
    _basemap(axm)
    order = ok.late_frac.argsort()
    sc = axm.scatter(ok.gauge_lon.iloc[order], ok.gauge_lat.iloc[order],
                     c=ok.late_frac.iloc[order], cmap="YlGnBu", vmin=0, vmax=1.0,
                     s=13, alpha=0.9, edgecolors="none", zorder=2, **tf)
    cb = plt.colorbar(sc, ax=axm, shrink=0.62, pad=0.02)
    cb.set_label("late fraction")
    axm.set_title(f"(b) Phase-shift index (share of response in spring+)\n({len(ok):,} catchments)")

    # (c) late-response fraction vs snow --------------------------------------
    axs = fig.add_subplot(gs[1, 1])
    x = ok.frac_snow.to_numpy(); y = ok.late_frac.to_numpy()
    axs.scatter(x, y, s=10, alpha=0.35, color="#4b8fd0", edgecolors="none")
    # binned medians
    ctr, med = [], []
    zero = x <= 0
    if zero.sum() >= 20:                       # the snow-free atom is its own point
        ctr.append(0.0); med.append(np.nanmedian(y[zero]))
    pos = x > 0
    if pos.sum() >= 40:                        # positive tail, equal-count bins
        edges = np.unique(np.nanquantile(x[pos], np.linspace(0, 1, 9)))
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            sel = pos & (x >= edges[i]) & ((x <= edges[i + 1]) if last else (x < edges[i + 1]))
            if sel.sum() >= 20:
                ctr.append(np.nanmedian(x[sel])); med.append(np.nanmedian(y[sel]))
    axs.plot(ctr, med, "-o", color="#c1666b", lw=2, ms=5, label="binned median (equal count)")
    rho = stats.spearmanr(x, y, nan_policy="omit").statistic
    axs.set_xlabel("snow fraction")
    axs.set_ylabel("late-response fraction")
    axs.set_title(f"(c) Late-response fraction against\nsnow fraction (Spearman $\\rho={rho:+.2f}$)")
    axs.legend(loc="lower right"); axs.grid(ls=":", alpha=0.4)

    # Explicit margins and no tight bbox: bbox_inches="tight" measures a cartopy
    # GeoAxes as empty and crops the map away, leaving only its colourbar.
    # Fixing the margins here also means the saved width is exactly FIG_W.
    fig.subplots_adjust(left=0.088, right=0.978, top=0.955, bottom=0.085)
    fig.savefig(out, dpi=300)
    pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {out} (+.pdf)  signal={len(ok)}"
          f"  [{_pdf_width_in(pdf):.2f} in wide]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", required=True)
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="figures/fig3_snowmelt_timing.png")
    a = ap.parse_args()
    betas = pd.read_parquet(a.signal).set_index("gauge_id")
    indices = pd.read_csv(a.indices).set_index("winter_year")
    spinup = pd.to_datetime(pd.read_csv(a.manifest).set_index("gauge_id")["spinup_end"])
    make(pd.read_parquet(a.timing), pd.read_parquet(a.attrs), betas, indices,
         a.states_dir, spinup, a.out)


if __name__ == "__main__":
    main()

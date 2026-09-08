#!/usr/bin/env python
"""
fig2_store_memory.py

Produces Figure 2 of the manuscript.

Two-reservoir memory figure -- the paper's differentiator. The HBV
filtering gives each store its own memory timescale tau (e-folding days of the
deseasonalised+detrended daily anomaly), and the two LONG-memory reservoirs are
physically distinct and geographically separated:

  * SP (snowpack)     -- seasonal cold-store memory, in snow-active catchments
                         (Alps, Nordics, uplands).
  * LZ (groundwater)  -- the slow aquifer store, with a heavy tail (chalk/karst
                         lowlands) that is the multi-annual memory in the sample.

Three panels: (a) distribution of tau per store on a log axis, ordered so the
two highlighted reservoirs SP and LZ sit together, with LZ's aquifer tail;
(b) map of snowpack memory tau_SP; (c) map of groundwater memory tau_LZ. The two
maps carry different colour scales, each matched to its own store. Maps reuse the
cartopy/fallback convention used for Figure 1.

    python figures/fig2_store_memory.py --strength-memory <response_strength_memory_DJF.parquet>

  With no --out, writes the published Figure 2 path (and the .pdf beside it).
  Run from the repository root, since that default path is relative to it.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

EXTENT = [-25, 32, 34, 72]          # lon0, lon1, lat0, lat1 (Europe)
SP_ACTIVE = 0.30                    # snowpack "active" if SP anomaly lives >=30% of winters
STORES = ["UZ", "Qsim", "SM", "SP", "LZ"]   # SP and LZ adjacent: the two long-memory reservoirs
STORE_LABEL = {"UZ": "UZ\n(upper/quick)", "Qsim": "Q\n(streamflow)",
               "SP": "SP\n(snowpack)", "SM": "SM\n(soil)", "LZ": "LZ\n(groundwater)"}
STORE_COLOR = {"UZ": "#9aa0a6", "Qsim": "#9aa0a6", "SP": "#4b8fd0",
               "SM": "#9aa0a6", "LZ": "#c1666b"}   # highlight the two reservoirs


def _basemap(ax):
    """Try cartopy coastlines/borders; return silently to a plain frame offline."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f1ec", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dce7ef", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#555", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.25, edgecolor="#999", zorder=1)
    except Exception as e:  # noqa: BLE001 - offline node etc.
        print(f"  (cartopy features unavailable: {type(e).__name__}; plain frame)", flush=True)


def _map_ax(fig, gs):
    """Make a map axis (cartopy if available, else plain lon/lat) + a transform kw."""
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


def _store_map(fig, gs, d, store, title, vmin, vmax, cblabel, log=True):
    ax, tf = _map_ax(fig, gs)
    _basemap(ax)
    t = d[f"tau_{store}"].clip(vmin, vmax)
    order = t.fillna(-1).argsort()             # draw long-memory points on top
    sc = ax.scatter(d.gauge_lon.iloc[order], d.gauge_lat.iloc[order], c=t.iloc[order],
                    cmap="viridis",
                    norm=(LogNorm(vmin=vmin, vmax=vmax) if log
                          else Normalize(vmin=vmin, vmax=vmax)),
                    s=13, alpha=0.9, edgecolors="none", zorder=2, **tf)
    cb = plt.colorbar(sc, ax=ax, shrink=0.62, pad=0.02, extend="both")
    cb.set_label(cblabel, fontsize=9)
    ax.set_title(f"{title}\n({len(d):,} catchments)", fontsize=10)
    return sc


def make(strength_memory: pd.DataFrame, out: str):
    d = strength_memory
    snow = d[(d.sp_active_frac >= SP_ACTIVE) & d.tau_SP.notna()].copy()
    gw = d[d.tau_LZ.notna()].copy()

    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.80, 1.4], hspace=0.02, wspace=0.10)

    # (a) distribution of tau per store, log axis --------------------------------
    axd = fig.add_subplot(gs[0, :])
    data, colors = [], []
    for s in STORES:
        v = (snow.tau_SP if s == "SP" else d[f"tau_{s}"]).dropna()
        data.append(v.to_numpy())
        colors.append(STORE_COLOR[s])
    bp = axd.boxplot(data, vert=True, widths=0.6, showfliers=False, patch_artist=True,
                     medianprops=dict(color="k", lw=1.4), whis=(5, 95))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#444")
    for i, v in enumerate(data, 1):
        axd.text(i, np.median(v), f"{np.median(v):.0f}d", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.55", lw=0.5, alpha=0.92))
    axd.set_yscale("log")
    axd.set_xticks(range(1, len(STORES) + 1))
    axd.set_xticklabels([STORE_LABEL[s] for s in STORES], fontsize=8.5)
    axd.set_ylabel(r"memory $\tau$ (days, log)", fontsize=9.5)
    axd.set_title("(a) Store memory timescales (box = IQR, whiskers 5–95th pct)",
                  fontsize=9.5, loc="left")
    axd.grid(axis="y", ls=":", alpha=0.4)

    # (b) snowpack memory map, (c) groundwater memory map ------------------------
    # The two maps carry different colour scales, each matched to its own store
    # (SP spans 47-97 d over 5-95 pct; LZ spans three orders of magnitude), so the
    # bars are labelled by store rather than both reading "memory tau".
    # SP spans only a factor of ~2.75, so a log scale buys nothing and prints
    # 4x10^1-style ticks; LZ spans three orders of magnitude and needs one.
    _store_map(fig, gs[1, 0], snow, "SP", "(b) Snowpack memory $\\tau_{SP}$",
               vmin=40, vmax=110, cblabel=r"$\tau_{SP}$ (days)", log=False)
    _store_map(fig, gs[1, 1], gw, "LZ", "(c) Groundwater memory $\\tau_{LZ}$",
               vmin=10, vmax=1100, cblabel=r"$\tau_{LZ}$ (days)")

    fig.savefig(out, dpi=300, bbox_inches="tight")
    import os
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+.pdf)  snow-active={len(snow)}  gw={len(gw)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength-memory", required=True)
    ap.add_argument("--out", default="figures/fig2_store_memory.png")
    a = ap.parse_args()
    make(pd.read_parquet(a.strength_memory), a.out)


if __name__ == "__main__":
    main()

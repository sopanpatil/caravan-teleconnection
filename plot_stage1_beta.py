#!/usr/bin/env python
"""
plot_stage1_beta.py

Map a Stage-1 sensitivity coefficient (default beta_NAO) across the catchment
sample. Points are coloured by beta on a symmetric diverging scale; FDR-significant
catchments (q<0.05) are drawn larger with a dark edge, non-significant ones smaller
and faded, so the map shows both the beta field and where it is trustworthy.

Coastlines/borders via cartopy when its Natural Earth data is reachable/cached;
otherwise the map falls back to a plain lon/lat frame (the 2,135 points trace
Europe well enough on their own).

    python plot_stage1_beta.py --stage1 <stage1_DJF.parquet> --index NAO --out <fig.png>
    python plot_stage1_beta.py --stage1 <...> --index all --outdir <figures/>
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTENT = [-25, 32, 36, 72]   # lon0, lon1, lat0, lat1 (Europe; data span -23.1..31.0, 37.3..70.9)
INDICES = ["NAO", "EA", "EAWR", "SCA"]


def _basemap(ax):
    """Try cartopy coastlines/borders; return True on success."""
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f1ec", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dce7ef", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.4, edgecolor="#555", zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.25, edgecolor="#999", zorder=1)
        return True
    except Exception as e:  # noqa: BLE001 - offline node etc.: fall back cleanly
        print(f"  (cartopy features unavailable: {type(e).__name__}; plain frame)", flush=True)
        return False


def plot_one(res: pd.DataFrame, idx: str, out: str):
    beta, sig = res[f"beta_{idx}"], res[f"sig_{idx}"]
    vmax = np.nanpercentile(np.abs(beta), 98)  # robust symmetric limits

    # figsize follows the extent's own aspect (57 deg lon x 36 deg lat), so the
    # map fills the canvas and the colourbar stays proportionate to it.
    on_map = False
    try:
        import cartopy.crs as ccrs
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(9.0, 6.4))
        ax = plt.axes(projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        tf = {"transform": proj}
        on_map = True
    except Exception:
        fig, ax = plt.subplots(figsize=(9.0, 6.4))
        ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1.4)
        tf = {}
    _basemap(ax)
    if on_map:
        try:
            gl = ax.gridlines(draw_labels=True, lw=0.3, color="#aaa",
                              alpha=0.6, linestyle=":")
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}
        except Exception as e:  # noqa: BLE001 - gridline labels need a recent cartopy
            print(f"  (gridlines unavailable: {type(e).__name__})", flush=True)

    ns = ~sig
    ax.scatter(res.gauge_lon[ns], res.gauge_lat[ns], c=beta[ns], cmap="coolwarm",
               vmin=-vmax, vmax=vmax, s=9, alpha=0.45, linewidths=0, zorder=2, **tf)
    sc = ax.scatter(res.gauge_lon[sig], res.gauge_lat[sig], c=beta[sig], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, s=30, alpha=0.95,
                    edgecolors="k", linewidths=0.3, zorder=3, **tf)

    cb = plt.colorbar(sc, ax=ax, shrink=0.72, aspect=34, pad=0.02, extend="both")
    cb.set_label(f"$\\beta_{{{idx}}}$  (mm day$^{{-1}}$ per unit index)", fontsize=10)
    n_sig = int(sig.sum())
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+.pdf)  vmax={vmax:.3f}  n_sig={n_sig}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--index", default="NAO", help="NAO|EA|EAWR|SCA|all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = pd.read_parquet(a.stage1)
    idxs = INDICES if a.index == "all" else [a.index]
    for idx in idxs:
        out = a.out if (a.out and a.index != "all") else os.path.join(a.outdir, f"stage1_beta_{idx}.png")
        plot_one(res, idx, out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
fig1_nao_precipitation_sensitivity.py

Produces Figure 1 of the manuscript.

Map a precipitation-sensitivity coefficient (default beta_NAO) across the catchment
sample. Points are coloured by beta on a symmetric diverging scale; FDR-significant
catchments (q<0.05) are drawn larger with a dark edge, non-significant ones smaller
and faded, so the map shows both the beta field and where it is trustworthy.

Coastlines/borders via cartopy when its Natural Earth data is reachable/cached;
otherwise the map falls back to a plain lon/lat frame (the 2,135 points trace
Europe well enough on their own).

    python figures/fig1_nao_precipitation_sensitivity.py --signal <precipitation_signal_DJF.parquet>
    python figures/fig1_nao_precipitation_sensitivity.py --signal <...> --index all --outdir <figures/>

  With no --out, writes the published Figure 1 path (and the .pdf beside it).
  Run from the repository root, since that default path is relative to it.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        fig = plt.figure(figsize=(FIG_W, 3.58))
        ax = plt.axes(projection=proj)
        ax.set_extent(EXTENT, crs=proj)
        tf = {"transform": proj}
        on_map = True
    except Exception:
        fig, ax = plt.subplots(figsize=(FIG_W, 3.58))
        ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1.4)
        tf = {}
    _basemap(ax)
    if on_map:
        try:
            gl = ax.gridlines(draw_labels=True, lw=0.3, color="#aaa",
                              alpha=0.6, linestyle=":")
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 9, "color": "#555"}
        except Exception as e:  # noqa: BLE001 - gridline labels need a recent cartopy
            print(f"  (gridlines unavailable: {type(e).__name__})", flush=True)

    ns = ~sig
    ax.scatter(res.gauge_lon[ns], res.gauge_lat[ns], c=beta[ns], cmap="coolwarm",
               vmin=-vmax, vmax=vmax, s=9, alpha=0.45, linewidths=0, zorder=2, **tf)
    sc = ax.scatter(res.gauge_lon[sig], res.gauge_lat[sig], c=beta[sig], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, s=30, alpha=0.95,
                    edgecolors="k", linewidths=0.3, zorder=3, **tf)

    cb = plt.colorbar(sc, ax=ax, shrink=0.72, aspect=34, pad=0.02, extend="both")
    cb.set_label(f"$\\beta_{{{idx}}}$  (mm day$^{{-1}}$ per unit index)")
    n_sig = int(sig.sum())
    # Explicit margins, and neither tight_layout nor a tight bbox. On an
    # aspect-locked GeoAxes, tight_layout resizes the map after the gridliner has
    # built its labels and cartopy >=0.25 then throws a GEOS error off the
    # degenerate boundary rings, while bbox_inches="tight" measures the GeoAxes
    # as empty and crops the map away, leaving only the colourbar. Fixing the
    # margins here also means the saved width is exactly FIG_W.
    fig.subplots_adjust(left=0.080, right=0.995, top=0.985, bottom=0.055)
    fig.savefig(out, dpi=300)
    pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  wrote {out} (+.pdf)  vmax={vmax:.3f}  n_sig={n_sig}"
          f"  [{_pdf_width_in(pdf):.2f} in wide]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--index", default="NAO", help="NAO|EA|EAWR|SCA|all")
    ap.add_argument("--out", default="figures/fig1_nao_precipitation_sensitivity.png")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = pd.read_parquet(a.signal)
    idxs = INDICES if a.index == "all" else [a.index]
    if a.index == "all":
        os.makedirs(a.outdir, exist_ok=True)
    for idx in idxs:
        out = a.out if (a.out and a.index != "all") else os.path.join(a.outdir, f"precipitation_sensitivity_{idx}.png")
        plot_one(res, idx, out)


if __name__ == "__main__":
    main()

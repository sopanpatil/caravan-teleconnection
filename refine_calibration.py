#!/usr/bin/env python
"""
refine_calibration.py

Downstream refinement of the aggregated calibration table
(`calibrated_parameters_ALL.csv`). Adds catchment attributes and two screens,
as flags rather than deletions so every decision is reversible:

  1. Glacier screen (uniform across ALL sources): flag catchments with
     HydroATLAS `gla_pc_sse` > GLACIER_MAX (%). Glacier-fed discharge is governed
     by ice mass-balance / melt rather than the seasonal P/PET forcing the
     teleconnection study targets. Applied uniformly so the screen does not
     depend on source (base Caravan applied it to non-Fennoscandia GRDC only;
     this re-screens the Fennoscandia glacier catchments it left in).

  2. Spatial de-duplication LamaH-CE <-> GRDC-Caravan. The two products overlap
     heavily in the Alpine / upper-Danube domain (same physical gauge archived
     twice). A catchment is a duplicate when a LamaH and a GRDC gauge sit within
     DEDUP_DIST_KM and their drainage areas agree to within DEDUP_AREA_RELDIFF.
     Both share identical Caravan (ERA5-Land) forcing + PET, so the choice does
     not affect forcing consistency; the LamaH-CE instance is retained as the
     curated regional product and the GRDC twin flagged as the duplicate.

Output columns added to the refined CSV:
  gauge_lat, gauge_lon, area, country, gla_pc_sse,
  glacier_pass  (gla_pc_sse <= GLACIER_MAX; NaN treated as pass),
  is_duplicate  (GRDC twin of a retained LamaH catchment),
  dup_partner   (the retained LamaH gauge_id, for duplicates),
  include_in_analysis  (calibration AND validation KGE > 0.5, and glacier_pass,
                        and not is_duplicate -- note this does NOT read
                        `used_in_analysis`, which is a looser upstream flag)

Also writes `dedup_pairs.csv` (the matched LamaH<->GRDC pairs with distance/area).
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

GLACIER_MAX = 5.0          # % catchment glacier cover (gla_pc_sse) screen threshold
DEDUP_DIST_KM = 1.0        # max gauge separation for a LamaH<->GRDC duplicate
DEDUP_AREA_RELDIFF = 0.15  # max |area difference| / area for a duplicate

# where each source's attribute files live, relative to --raw
ATTR_GLOB = {
    "camelsdk": "DK_extracted/attributes/camelsdk",
    "camelsgb": "base_extracted/Caravan-nc/attributes/camelsgb",
    "lamah":    "base_extracted/Caravan-nc/attributes/lamah",
    "grdc":     "GRDC_extracted/GRDC-Caravan-extension-nc/attributes/grdc",
}


def _read_first_col_id(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={df.columns[0]: "gauge_id"})


def load_attributes(raw: str) -> pd.DataFrame:
    """One row per gauge_id: lat/lon/area/country (+ gla_pc_sse) for all sources."""
    parts = []
    for src, sub in ATTR_GLOB.items():
        d = os.path.join(raw, sub)
        other = glob.glob(os.path.join(d, "attributes_other_*.csv"))
        hydro = glob.glob(os.path.join(d, "attributes_hydroatlas_*.csv"))
        if not other or not hydro:
            raise FileNotFoundError(f"missing attribute CSVs for {src} under {d}")
        o = _read_first_col_id(other[0])[["gauge_id", "gauge_lat", "gauge_lon", "area", "country"]]
        h = _read_first_col_id(hydro[0])[["gauge_id", "gla_pc_sse"]]
        parts.append(o.merge(h, on="gauge_id", how="left"))
    return pd.concat(parts, ignore_index=True)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Match each LamaH catchment to its nearest GRDC catchment; keep true dupes."""
    lam = df[(df.source == "lamah")].dropna(subset=["gauge_lat", "gauge_lon", "area"])
    grd = df[(df.source == "grdc")].dropna(subset=["gauge_lat", "gauge_lon", "area"])
    g_lat, g_lon, g_area = grd.gauge_lat.values, grd.gauge_lon.values, grd.area.values
    g_id = grd.gauge_id.values
    rows = []
    for r in lam.itertuples(index=False):
        d = haversine_km(r.gauge_lat, r.gauge_lon, g_lat, g_lon)
        j = int(np.argmin(d))
        reldiff = abs(r.area - g_area[j]) / max(r.area, 1e-9)
        if d[j] < DEDUP_DIST_KM and reldiff < DEDUP_AREA_RELDIFF:
            rows.append((r.gauge_id, g_id[j], round(float(d[j]), 4),
                         round(float(r.area), 2), round(float(g_area[j]), 2),
                         round(float(reldiff), 4)))
    return pd.DataFrame(rows, columns=["lamah_id", "grdc_id", "dist_km",
                                       "lamah_area", "grdc_area", "area_reldiff"])


def refine(cal: pd.DataFrame, attrs: pd.DataFrame):
    df = cal.merge(attrs, on="gauge_id", how="left")

    df["glacier_pass"] = ~(df["gla_pc_sse"] > GLACIER_MAX)  # NaN -> not >thr -> pass

    pairs = find_duplicates(df)
    dup_grdc = dict(zip(pairs.grdc_id, pairs.lamah_id))
    df["is_duplicate"] = df["gauge_id"].isin(dup_grdc)
    df["dup_partner"] = df["gauge_id"].map(dup_grdc)

    # Analysis screen, on the non-glacier, non-duplicate set: BOTH calibration and
    # validation KGE > 0.5. The memory/timing metrics are read from the model's internal
    # stores, so we keep only catchments whose store dynamics are constrained by a fit
    # that reproduces observed discharge well in BOTH periods; a symmetric two-period
    # threshold is the most defensible cut (and a basin can validate well yet fail in
    # calibration, leaving its store split untrustworthy). This also excludes the
    # coverage-gap basins whose calibration window held no observations (SCE-UA returned
    # the -1e6 sentinel).
    df["include_in_analysis"] = (
        (df["calibration_kge"] > 0.5)
        & (df["validation_kge"] > 0.5)
        & df["glacier_pass"]
        & ~df["is_duplicate"]
    )
    return df, pairs


def summarise(df: pd.DataFrame, pairs: pd.DataFrame):
    print(f"de-duplication: {len(pairs)} LamaH<->GRDC pairs "
          f"(dist<{DEDUP_DIST_KM}km, area reldiff<{DEDUP_AREA_RELDIFF})", flush=True)
    if len(pairs):
        by = df[df.is_duplicate].merge(pairs, left_on="gauge_id", right_on="grdc_id")
        print("  dropped GRDC twins by country:", flush=True)
        print(df[df.is_duplicate].country.value_counts().to_string(), flush=True)
    gl = df[~df.glacier_pass]
    print(f"\nglacier screen (gla_pc_sse>{GLACIER_MAX}%): {len(gl)} flagged", flush=True)
    print(gl.groupby("source").size().to_string(), flush=True)
    print("\nper-source: used_in_analysis -> include_in_analysis", flush=True)
    for src, g in df.groupby("source"):
        print(f"  {src:10s} used={int(g.used_in_analysis.sum()):4d}  "
              f"-glacier={int((g.used_in_analysis & ~g.glacier_pass).sum()):3d}  "
              f"-dupe={int((g.used_in_analysis & g.glacier_pass & g.is_duplicate).sum()):4d}  "
              f"=> include={int(g.include_in_analysis.sum()):4d}", flush=True)
    print(f"\nTOTAL used={int(df.used_in_analysis.sum())}  "
          f"=> include={int(df.include_in_analysis.sum())}", flush=True)


def selftest():
    # two coincident LamaH/GRDC gauges (dupe), one distant, one glacierised LamaH
    cal = pd.DataFrame({
        "gauge_id": ["lamah_A", "GRDC_A", "lamah_B", "GRDC_B", "lamah_G"],
        "source":   ["lamah", "grdc", "lamah", "grdc", "lamah"],
        "used_in_analysis": [True, True, True, True, True],
        # lamah_B validates below the 0.5 screen (tests the validation-KGE cut);
        # GRDC_B has a sentinel calibration (coverage gap) despite val>0.5 (tests CAL_RAN)
        "validation_kge":   [0.70, 0.60, 0.40, 0.60, 0.70],
        "calibration_kge":  [0.60, 0.50, 0.50, -1e6, 0.60],
    })
    attrs = pd.DataFrame({
        "gauge_id": ["lamah_A", "GRDC_A", "lamah_B", "GRDC_B", "lamah_G"],
        "gauge_lat": [47.0, 47.0005, 48.0, 49.0, 47.5],
        "gauge_lon": [12.0, 12.0005, 13.0, 14.0, 12.5],
        "area":      [100.0, 101.0, 200.0, 500.0, 50.0],
        "country":   ["Austria"] * 5,
        "gla_pc_sse": [0.0, 0.0, 1.0, 2.0, 42.0],
    })
    df, pairs = refine(cal, attrs)
    assert len(pairs) == 1 and pairs.iloc[0].grdc_id == "GRDC_A", pairs
    assert bool(df.set_index("gauge_id").loc["GRDC_A", "is_duplicate"]) is True
    assert bool(df.set_index("gauge_id").loc["lamah_A", "include_in_analysis"]) is True
    assert bool(df.set_index("gauge_id").loc["GRDC_A", "include_in_analysis"]) is False
    assert bool(df.set_index("gauge_id").loc["lamah_G", "include_in_analysis"]) is False  # glacier
    assert bool(df.set_index("gauge_id").loc["lamah_B", "include_in_analysis"]) is False  # val KGE < 0.5
    assert bool(df.set_index("gauge_id").loc["GRDC_B", "include_in_analysis"]) is False  # sentinel calibration
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-csv", help="calibrated_parameters_ALL.csv")
    ap.add_argument("--raw", help="caravan_raw root holding the attribute trees")
    ap.add_argument("--out", help="output refined CSV path")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.all_csv and a.raw and a.out):
        ap.error("--all-csv, --raw and --out are required unless --selftest")

    cal = pd.read_csv(a.all_csv)
    attrs = load_attributes(a.raw)
    df, pairs = refine(cal, attrs)
    df.to_csv(a.out, index=False)
    pairs.to_csv(os.path.join(os.path.dirname(a.out), "dedup_pairs.csv"), index=False)
    summarise(df, pairs)
    print(f"\nwrote {a.out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()

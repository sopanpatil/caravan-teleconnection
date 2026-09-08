#!/usr/bin/env python
"""
build_seasonal_table.py

Input builder for the response analyses. Collapses the daily obs-forced HBV state parquets
(generate_states.py) into per-catchment SEASONAL means + anomalies keyed by
winter-year, then joins the teleconnection indices (fetch_teleconnection_indices.py).

Season convention is copied EXACTLY from fetch_teleconnection_indices.py so the
catchment seasons line up with the index seasons:
  DJF     winter_year Y = mean(Dec(Y-1), Jan(Y), Feb(Y))   (December rolled +1)
A winter is dropped for a variable unless it has >= --min-days finite days of it.

Spin-up: each catchment is trimmed to dates >= its manifest `spinup_end`
(first forcing day + 5 yr) BEFORE aggregation, so early LZ (an initial-condition
artefact) enters neither the seasonal values nor the climatology used for anomalies.

Anomalies are per-catchment: value minus that catchment's own mean over its
available post-spin-up winters (fit_precipitation_signal.py exploits within-catchment temporal
variability, so a per-catchment baseline is the right reference).

Output (one row per catchment x winter_year):
  gauge_id, source, winter_year,
  <var>_<season>, <var>_<season>_anom   for var in VARS,
  + all index columns joined on winter_year.
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd

# variables carried through: precipitation-signal driver (precip), stores (SM/SP/UZ/LZ),
# flow response (Qsim), snow flux (melt), covariates (temp/pet), validation (flow_obs)
VARS = ["precip", "temp", "pet", "melt", "Qsim", "flow_obs", "SM", "SP", "UZ", "LZ"]

# DJF only: the paper analyses boreal winter, and no other season was run,
# so no other is offered here. seasonal_catchment() itself stays general.
SEASONS = {
    "DJF": ([12, 1, 2], {12}),
}

# A winter needs this many finite daily values of a variable to yield a mean.
DEFAULT_MIN_DAYS = 80


def seasonal_catchment(df: pd.DataFrame, months, rollover, min_days: int) -> pd.DataFrame:
    """Daily frame (DatetimeIndex) -> winter_year-indexed seasonal means + anoms."""
    d = df[df.index.month.isin(months)]
    if d.empty:
        return pd.DataFrame()
    wy = d.index.year + d.index.month.isin(rollover).astype(int)
    g = d[VARS].groupby(wy)
    means = g.mean()
    means = means.where(g.count() >= min_days)          # per-var coverage gate
    means = means.dropna(how="all")
    if means.empty:
        return means
    anom = means - means.mean(axis=0)                    # per-catchment climatology
    out = means.join(anom.add_suffix("_anom"))
    out.index.name = "winter_year"
    return out


def build(states_dir, manifest_path, indices_path, season, min_days):
    months, rollover = SEASONS[season]
    man = pd.read_csv(manifest_path).set_index("gauge_id")
    spinup = pd.to_datetime(man["spinup_end"])
    idx = pd.read_csv(indices_path)

    parts = []
    files = sorted(glob.glob(os.path.join(states_dir, "*", "*.parquet")))
    for i, f in enumerate(files, 1):
        gid = os.path.basename(f)[:-len(".parquet")]
        src = os.path.basename(os.path.dirname(f))
        df = pd.read_parquet(f)
        if gid in spinup.index:
            df = df[df.index >= spinup.loc[gid]]
        seas = seasonal_catchment(df, months, rollover, min_days)
        if seas.empty:
            continue
        seas = seas.rename(columns=lambda c: f"{c}_{season}" if not c.endswith("_anom")
                           else c.replace("_anom", f"_{season}_anom"))
        seas.insert(0, "source", src)
        seas.insert(0, "gauge_id", gid)
        parts.append(seas.reset_index())
        if i % 500 == 0:
            print(f"  {i}/{len(files)} catchments aggregated", flush=True)

    panel = pd.concat(parts, ignore_index=True)
    merged = panel.merge(idx, on="winter_year", how="left")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--indices", default="teleconnection_seasonal.csv")
    ap.add_argument("--out")
    ap.add_argument("--season", default="DJF", choices=list(SEASONS))
    ap.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.states_dir and a.manifest and a.out):
        ap.error("--states-dir, --manifest and --out are required unless --selftest")

    merged = build(a.states_dir, a.manifest, a.indices, a.season, a.min_days)
    merged.to_parquet(a.out)
    n_cat = merged.gauge_id.nunique()
    wy = merged.winter_year
    print(f"wrote {a.out}", flush=True)
    print(f"  rows={len(merged)}  catchments={n_cat}  winter_year {wy.min()}-{wy.max()}", flush=True)
    print(f"  NAO_DJF non-null rows: {merged['NAO_DJF'].notna().sum()}", flush=True)


def selftest():
    # two winters of synthetic daily data; DJF rollover + anomaly must be exact
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    df = pd.DataFrame({v: 1.0 for v in VARS}, index=dates)
    # make DJF 2001 (Dec2000+Jan/Feb2001) precip=3, DJF 2000 (Jan/Feb2000 only) precip=1
    df.loc[df.index.month.isin([12]) & (df.index.year == 2000), "precip"] = 3.0
    df.loc[(df.index.month.isin([1, 2])) & (df.index.year == 2001), "precip"] = 3.0
    seas = seasonal_catchment(df, *SEASONS["DJF"], min_days=50)
    # winter 2000 has only Jan+Feb (~59 days) -> present; winter 2001 full
    assert 2001 in seas.index, seas.index.tolist()
    assert abs(seas.loc[2001, "precip"] - 3.0) < 1e-9
    # anomaly must be value minus mean across the catchment's winters
    m = seas["precip"].mean()
    assert abs(seas.loc[2001, "precip_anom"] - (3.0 - m)) < 1e-9
    # coverage gate: raise min_days above a partial winter's day-count -> dropped
    seas2 = seasonal_catchment(df, *SEASONS["DJF"], min_days=85)
    assert 2000 not in seas2.index  # Jan+Feb 2000 (~59 d) < 85 -> gated out
    print("selftest OK  (DJF rollover, coverage gate, per-catchment anomaly all correct)")


if __name__ == "__main__":
    main()

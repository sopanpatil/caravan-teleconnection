#!/usr/bin/env python
"""
generate_states.py

Observation-forced historical HBV run for the teleconnection analysis. For every
catchment flagged `include_in_analysis` in the refined
calibration table, force the *calibrated* HBV with that catchment's Caravan
(ERA5-Land) P/T/PET and save the full DAILY state series.

Daily resolution is deliberate: Stage-2 estimates a memory timescale tau per
store via lagged cross-correlation (tau(SM) ~ weeks, tau(LZ) ~ months-years),
which cannot be recovered from pre-aggregated seasonal data. Seasonal/monthly
aggregation for Stage-1 is derived downstream from these daily files.

Per catchment we write one parquet with the date index plus:
  forcing : precip, temp, pet, flow_obs   (flow_obs = observed streamflow, mm/day)
  flux    : Qsim (MAXBAS-routed), Qgen, Q0, Q1, Q2, ETact, melt
  stores  : SM, SP, UZ, LZ

Spin-up is NOT trimmed here (kept reversible): the model runs from the start of
each catchment's forcing and the manifest records `spinup_end` = first date +
SPINUP_YEARS. Downstream analysis should start no earlier than that so early LZ
is not an initial-condition artefact; ERA5-Land starts 1981 so the common
post-spin-up window is ~1986-2020.

Usage:
    python generate_states.py --refined <ALL_refined.csv> --raw <caravan_raw> \
        --out <states_dir> [--limit N] [--sources camelsgb,lamah]
    python generate_states.py --selftest
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

from caravan_io import load_basin

# Set HBV_MODEL_REPO to a checkout of https://github.com/sopanpatil/hbv-model,
# or leave it unset and place that checkout alongside this repository.
_HBV_REPO = os.environ.get(
    "HBV_MODEL_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hbv-model"),
)
if _HBV_REPO not in sys.path:
    sys.path.insert(0, _HBV_REPO)
from hbv_model.hbv import HBVModel  # noqa: E402

SPINUP_YEARS = 5

PARAM_NAMES = ["TT", "CFMAX", "CFR", "CWH", "FC", "LP", "BETA",
               "K0", "K1", "K2", "UZL", "PERC", "MAXBAS"]

# per-source extracted-dataset root(s), relative to --raw. A source may span
# several extraction dirs (GRDC was extracted in two disjoint tar passes:
# Fennoscandia -> GRDC_extracted, the rest of Europe -> GRDC_euro_extracted), so
# each maps to a LIST of candidate roots tried in order.
SOURCE_ROOT = {
    "camelsdk": ["DK_extracted"],
    "camelsgb": ["base_extracted/Caravan-nc"],
    "lamah":    ["base_extracted/Caravan-nc"],
    "grdc":     ["GRDC_extracted/GRDC-Caravan-extension-nc",
                 "GRDC_euro_extracted/GRDC-Caravan-extension-nc"],
}


def resolve_nc(raw, src, gid):
    """First existing <root>/timeseries/netcdf/<src>/<gid>.nc across candidate roots."""
    candidates = [os.path.join(raw, root, "timeseries", "netcdf", src, f"{gid}.nc")
                  for root in SOURCE_ROOT[src]]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # non-existent; caller raises a clear FileNotFoundError

# columns saved, in order
FORCING_COLS = ["precip", "temp", "pet", "flow_obs"]
STATE_COLS = ["Qsim", "Qgen", "Q0", "Q1", "Q2", "ETact", "melt", "SM", "SP", "UZ", "LZ"]


def run_catchment(nc_path, params: dict) -> pd.DataFrame:
    """Force calibrated HBV with a basin's Caravan forcing -> daily state frame."""
    bf = load_basin(nc_path)
    q_routed, states = HBVModel(params).run(bf.precip, bf.temp, bf.pet)
    data = {
        "precip": bf.precip, "temp": bf.temp, "pet": bf.pet, "flow_obs": bf.flow,
        "Qsim": q_routed, "Qgen": states["Qgen"],
        "Q0": states["Q0"], "Q1": states["Q1"], "Q2": states["Q2"],
        "ETact": states["ETact"], "melt": states["melt"],
        "SM": states["SM"], "SP": states["SP"], "UZ": states["UZ"], "LZ": states["LZ"],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(bf.dates, name="date"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", help="calibrated_parameters_ALL_refined.csv")
    ap.add_argument("--raw", help="caravan_raw root")
    ap.add_argument("--out", help="output states dir")
    ap.add_argument("--sources", default=None, help="comma list to restrict (default all)")
    ap.add_argument("--limit", type=int, default=None, help="cap #catchments (testing)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.refined and a.raw and a.out):
        ap.error("--refined, --raw and --out are required unless --selftest")

    df = pd.read_csv(a.refined)
    sub = df[df.include_in_analysis].copy()
    if a.sources:
        keep = set(a.sources.split(","))
        sub = sub[sub.source.isin(keep)]
    if a.limit:
        sub = sub.head(a.limit)
    print(f"catchments to generate: {len(sub)}", flush=True)

    manifest = []
    done = skipped = failed = 0
    for i, row in enumerate(sub.itertuples(index=False), 1):
        src, gid = row.source, row.gauge_id
        out_dir = os.path.join(a.out, src)
        os.makedirs(out_dir, exist_ok=True)
        outfile = os.path.join(out_dir, f"{gid}.parquet")
        if os.path.exists(outfile):
            skipped += 1
            continue
        # every source names its file <gauge_id>.nc; resolve_nc searches the
        # source's candidate roots. (Not caravan_io.basin_nc_path: its prefix check
        # is case-sensitive and double-prefixes GRDC's 'GRDC_' ids under 'grdc/'.)
        nc = resolve_nc(a.raw, src, gid)
        try:
            params = {k: getattr(row, k) for k in PARAM_NAMES}
            out = run_catchment(nc, params)
            out.to_parquet(outfile)
            manifest.append({
                "gauge_id": gid, "source": src, "n_days": len(out),
                "date_start": out.index.min().date(), "date_end": out.index.max().date(),
                "spinup_end": (out.index.min() + pd.DateOffset(years=SPINUP_YEARS)).date(),
                "SM_finite": float(np.isfinite(out.SM).mean()),
                "LZ_finite": float(np.isfinite(out.LZ).mean()),
                "flow_obs_finite": float(np.isfinite(out.flow_obs).mean()),
            })
            done += 1
        except Exception as e:  # noqa: BLE001 - keep the batch alive, log the basin
            print(f"  FAIL {src}/{gid}: {type(e).__name__}: {e}", flush=True)
            failed += 1
        if i % 250 == 0:
            print(f"  {i}/{len(sub)}  done={done} skip={skipped} fail={failed}", flush=True)

    if manifest:
        man = pd.DataFrame(manifest)
        man_path = os.path.join(a.out, "states_manifest.csv")
        # append if resuming
        if os.path.exists(man_path):
            man = pd.concat([pd.read_csv(man_path), man], ignore_index=True) \
                    .drop_duplicates("gauge_id", keep="last")
        man.to_csv(man_path, index=False)
        print(f"manifest -> {man_path} ({len(man)} rows)", flush=True)
    print(f"DONE  generated={done}  skipped={skipped}  failed={failed}", flush=True)


def selftest():
    # deterministic synthetic forcing -> states must be finite, mass-plausible
    n = 4000
    rng = np.random.default_rng(0)
    precip = np.clip(rng.gamma(0.4, 6.0, n), 0, None)
    temp = 10 + 8 * np.sin(np.arange(n) * 2 * np.pi / 365.25) + rng.normal(0, 2, n)
    pet = np.clip(2 + 2 * np.sin(np.arange(n) * 2 * np.pi / 365.25), 0, None)
    dates = pd.date_range("1981-01-01", periods=n, freq="D")

    params = dict(TT=0.0, CFMAX=3.0, CFR=0.05, CWH=0.1, FC=250.0, LP=0.6, BETA=2.0,
                  K0=0.3, K1=0.1, K2=0.02, UZL=20.0, PERC=2.0, MAXBAS=3.0)
    q_routed, states = HBVModel(params).run(precip, temp, pet)
    for k in ("SM", "SP", "UZ", "LZ", "Qgen"):
        assert np.all(np.isfinite(states[k])), f"{k} not all finite"
    assert np.all(states["SM"] >= -1e-9) and np.all(states["SM"] <= params["FC"] + 1e-6)
    assert np.all(states["SP"] >= -1e-9) and np.all(states["LZ"] >= -1e-9)
    # flux closure: Qgen == Q0+Q1+Q2
    closure = np.max(np.abs(states["Qgen"] - (states["Q0"] + states["Q1"] + states["Q2"])))
    assert closure < 1e-9, f"flux closure off by {closure}"
    print("selftest OK  (states finite, bounded, flux closes; Qsim mean="
          f"{np.nanmean(q_routed):.3f} mm/day)")


if __name__ == "__main__":
    main()

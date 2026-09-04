#!/usr/bin/env python
"""
calibrate_single_caravan.py

Single-catchment HBV calibration worker for Caravan-format basins, using the
published hbv-model repo (HBVModel + calibrate_sceua). Designed to be driven
one basin at a time by a job array (analogous to
rainfallrunoff/calibrate_single.py, but reading Caravan netCDF via caravan_io
and calibrating against ERA5-Land forcing + the dataset's FAO-PM PET).

Because Caravan uses ERA5-Land forcing, the GB `calibrated_parameters.csv`
(tuned on native CAMELS-GB forcing) is NOT valid here -- every catchment,
GB included, must be re-calibrated on Caravan forcing. This worker is that step.

Usage:
    python calibrate_single_caravan.py <source> <nc_path_or_gauge_id> [--root DIR] [--out DIR]

Writes one JSON per basin: {gauge_id, source, calibration_kge, validation_kge,
n_cal_obs, n_val_obs, params}. Skips basins whose output already exists, so a
failed array can be safely re-submitted.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

from caravan_io import load_basin, basin_nc_path

# Make the published hbv-model importable without installing it (no pyproject).
# Set HBV_MODEL_REPO to a checkout of https://github.com/sopanpatil/hbv-model,
# or leave it unset and place that checkout alongside this repository.
_HBV_REPO = os.environ.get(
    "HBV_MODEL_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hbv-model"),
)
if _HBV_REPO not in sys.path:
    sys.path.insert(0, _HBV_REPO)
from hbv_model.hbv import HBVModel          # noqa: E402
from hbv_model import calibrate_sceua        # noqa: E402
from hbv_model import metrics as metrics_module  # noqa: E402

# --- Split (model runs full 1981-2020; >=2 yr warmup before scoring) ---------
CAL_START, CAL_END = "1983-01-01", "2002-12-31"
VAL_START, VAL_END = "2003-01-01", "2020-12-31"

# --- SCE-UA settings (match the GB calibration precedent) --------------------
N_COMPLEXES = int(os.environ.get("SCEUA_NCOMPLEXES", 7))
MAXITER = int(os.environ.get("SCEUA_MAXITER", 15000))
SEED = int(os.environ.get("SCEUA_SEED", 42))


def _n_obs(flow, dates, start, end):
    import pandas as pd
    d = pd.to_datetime(dates)
    m = ~np.isnan(flow) & (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return int(m.sum())


def calibrate_basin(nc_path, source, out_dir, *, verbose=False):
    bf = load_basin(nc_path)
    gauge_id = bf.gauge_id
    outfile = os.path.join(out_dir, f"{gauge_id}_hbv_params.json")
    if os.path.exists(outfile):
        print(f"SKIP {gauge_id}: output exists")
        return

    best_params, cal_kge = calibrate_sceua(
        model_cls=HBVModel,
        precip=bf.precip, temp=bf.temp, evap=bf.pet, q_obs=bf.flow,
        dates=bf.dates, start_date=CAL_START, end_date=CAL_END,
        metric="kge", n_complexes=N_COMPLEXES, maxiter=MAXITER,
        seed=SEED, verbose=verbose,
    )

    # Validation KGE on the held-out window with the calibrated params.
    q_sim, _ = HBVModel(best_params).run(bf.precip, bf.temp, bf.pet)
    val_kge = metrics_module.kge(
        bf.flow, q_sim, dates=bf.dates, start_date=VAL_START, end_date=VAL_END
    )

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "gauge_id": gauge_id,
        "source": source,
        "calibration_kge": float(cal_kge),
        "validation_kge": float(val_kge) if np.isfinite(val_kge) else None,
        "n_cal_obs": _n_obs(bf.flow, bf.dates, CAL_START, CAL_END),
        "n_val_obs": _n_obs(bf.flow, bf.dates, VAL_START, VAL_END),
        "params": {k: float(v) for k, v in best_params.items()},
    }
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{gauge_id}: cal_KGE={cal_kge:.4f}  val_KGE={val_kge:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Caravan source dir, e.g. camelsdk / camelsgb / grdc")
    ap.add_argument("target", help="a .nc path, or a gauge_id (needs --root)")
    ap.add_argument("--root", default=None, help="extracted dataset root (for gauge_id lookup)")
    ap.add_argument("--out", default="calibrated_params_caravan", help="output dir")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    nc = a.target if a.target.endswith(".nc") else str(basin_nc_path(a.root, a.source, a.target))
    out_dir = os.path.join(a.out, a.source)
    os.makedirs(out_dir, exist_ok=True)
    calibrate_basin(nc, a.source, out_dir, verbose=a.verbose)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
run_calibration_batch.py

Calibrate every basin of a Caravan source in one process (one numba compile,
no cache races), by looping calibrate_catchment.calibrate_basin. Resumable
(skips basins whose JSON already exists) and fault-tolerant (a failing basin is
logged and skipped, not fatal). Suitable for a long-running background job for the
smaller national sets (e.g. DK); the same worker also drives a job array for
the large sets.

Usage:
    python run_calibration_batch.py <source> <netcdf_dir> [--out DIR]
"""
from __future__ import annotations
import argparse
import glob
import os
import time

import numpy as np

from calibrate_catchment import calibrate_basin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="e.g. camelsdk")
    ap.add_argument("netcdf_dir", help="dir of <source>_*.nc files")
    ap.add_argument("--out", default="calibrated_params_caravan")
    a = ap.parse_args()

    out_dir = os.path.join(a.out, a.source)
    os.makedirs(out_dir, exist_ok=True)
    # glob all .nc in the dir (the source LABEL may differ in case from the file
    # prefix, e.g. label 'grdc' but files 'GRDC_*.nc'); the dir holds only basins.
    files = sorted(glob.glob(os.path.join(a.netcdf_dir, "*.nc")))
    print(f"[{a.source}] {len(files)} basins -> {out_dir}", flush=True)

    t0 = time.time()
    done = fail = 0
    for i, f in enumerate(files, 1):
        try:
            calibrate_basin(f, a.source, out_dir)
            done += 1
        except Exception as e:  # noqa: BLE001 - log and continue the batch
            fail += 1
            print(f"FAIL {os.path.basename(f)}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  progress {i}/{len(files)}  ok={done} fail={fail}  "
                  f"elapsed={el/60:.1f} min  ({el/i:.1f} s/basin)", flush=True)

    print(f"[{a.source}] DONE ok={done} fail={fail} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

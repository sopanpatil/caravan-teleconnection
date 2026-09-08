#!/usr/bin/env python
"""
aggregate_calibration.py

Collect the per-basin calibration JSONs (one dir per source under --out) into
per-source CSVs plus a combined CSV, mirroring the GB calibrated_parameters.csv
layout. `used_in_analysis` is a permissive first pass, flagging basins with KGE
>= 0.3 in both calibration and validation. It is NOT the analysis screen: that is
applied downstream by screen_analysis_sample.py as `include_in_analysis` (KGE > 0.5
in BOTH periods, plus the glacier and de-duplication screens), computed from the
KGE columns directly rather than from this flag. Nothing downstream of
screen_analysis_sample.py reads `used_in_analysis`.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

KGE_FLOOR = 0.3


def load_source(d: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        row = {
            "gauge_id": r["gauge_id"], "source": r["source"],
            "calibration_kge": r["calibration_kge"], "validation_kge": r["validation_kge"],
            "n_cal_obs": r.get("n_cal_obs"), "n_val_obs": r.get("n_val_obs"),
        }
        row.update(r["params"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir holding per-source subdirs of JSONs")
    a = ap.parse_args()

    parts = []
    for src in sorted(os.listdir(a.out)):
        d = os.path.join(a.out, src)
        if not os.path.isdir(d):
            continue
        df = load_source(d)
        if df.empty:
            continue
        df["used_in_analysis"] = (
            (df["calibration_kge"] >= KGE_FLOOR)
            & (df["validation_kge"].fillna(-9) >= KGE_FLOOR)
        )
        df.to_csv(os.path.join(a.out, f"calibrated_parameters_{src}.csv"), index=False)
        parts.append(df)
        cal, val = df["calibration_kge"], df["validation_kge"].dropna()
        print(f"{src:10s} n={len(df):4d}  used={int(df['used_in_analysis'].sum()):4d}  "
              f"cal_KGE med={cal.median():.3f}  val_KGE med={val.median():.3f}", flush=True)

    if parts:
        comb = pd.concat(parts, ignore_index=True)
        comb.to_csv(os.path.join(a.out, "calibrated_parameters_ALL.csv"), index=False)
        print(f"TOTAL       n={len(comb):4d}  used={int(comb['used_in_analysis'].sum()):4d}", flush=True)


if __name__ == "__main__":
    main()

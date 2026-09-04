#!/usr/bin/env python
"""
verify_text_s2.py

Regression check on the numbers quoted in Text S2 of the manuscript ("Memory and the
duration of the forced response"), regenerated from the ARCHIVED derived products alone
-- no state files, no re-run of the lag profiles.

Text S2 is the test that MEMORY, which is estimated from the flow series alone and never
refers to the fitted forcing Phat', nevertheless governs how long a teleconnection
anomaly persists. It is the claim that licenses treating memory as one of the three
response properties, and memory is the only property whose physiographic controls
survive the wild cluster bootstrap, so the paper leans on it. This script exists so that
the claim cannot drift silently if the upstream stages are re-run.

Subsetting criterion, which is easy to get wrong: "snow-free" here is the HydroATLAS
attribute frac_snow <= 0.05, restricted to catchments with a detectable signal in BOTH
the observed and simulated lag profiles (sig > 0.2 in each). The validate() print inside
stage2c_registration_lag.py / stage2_obs_validation.py instead uses the MODEL's
sp_active_frac, which is what those scripts have to hand: same sign and monotonicity,
different n and rho. Do not cross-quote the two.

    python verify_text_s2.py --derived derived_data
    python verify_text_s2.py --derived <dir> --quiet    # exit code only
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

SIG_MIN = 0.2        # peak |r| needed before a lag profile is usable, as Stage-2c
SNOW_FREE = 0.05     # HydroATLAS frac_snow at or below this counts as snow-free

# (label, value quoted in Text S2, tolerance)
EXPECTED = {
    "n_snow_free":   (955,    2),
    "rho_tau_obs":   (+0.43,  0.02),
    "rho_tau_LZ":    (+0.44,  0.02),
    "rho_all":       (+0.22,  0.02),
    "q1_tau_obs":    (0.09,   0.02),
    "q4_tau_obs":    (0.30,   0.03),
    "q1_tau_LZ":     (0.07,   0.02),
    "q4_tau_LZ":     (0.33,   0.03),
}


def load(derived: str) -> pd.DataFrame:
    d = derived.rstrip("/")
    obs = pd.read_parquet(f"{d}/stage2_obs_DJF.parquet").set_index("gauge_id")
    c2c = pd.read_parquet(f"{d}/stage2c_DJF.parquet").set_index("gauge_id")
    s2 = pd.read_parquet(f"{d}/stage2_DJF.parquet").set_index("gauge_id")
    at = pd.read_parquet(f"{d}/attributes.parquet").set_index("gauge_id")
    if "retention_obs" not in obs.columns:
        sys.exit("stage2_obs_DJF.parquet has no retention_obs column: re-run "
                 "stage2_obs_validation.py (the metric was added 2026-09-03)")
    t = (obs[["retention_obs", "tau_Qobs", "sig_obs"]]
         .join(c2c[["sig"]]).join(s2[["tau_LZ"]]).join(at[["frac_snow"]]))
    return t[(t.sig_obs > SIG_MIN) & (t.sig > SIG_MIN)]


def quartile_medians(g: pd.DataFrame, col: str) -> pd.Series:
    q = pd.qcut(g[col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return g.groupby(q, observed=True).retention_obs.median()


def compute(t: pd.DataFrame) -> dict:
    sf = t[t.frac_snow <= SNOW_FREE]
    out = {"n_signal": len(t), "n_snow_free": len(sf)}
    for key, col in [("tau_obs", "tau_Qobs"), ("tau_LZ", "tau_LZ")]:
        g = sf[["retention_obs", col]].dropna()
        g = g[g[col] > 0]
        out[f"rho_{key}"] = stats.spearmanr(g.retention_obs, np.log(g[col])).statistic
        out[f"n_{key}"] = len(g)
        med = quartile_medians(g, col)
        out[f"q1_{key}"], out[f"q4_{key}"] = float(med.iloc[0]), float(med.iloc[-1])
        out[f"quartiles_{key}"] = [round(float(v), 3) for v in med]
    g = t[["retention_obs", "tau_Qobs"]].dropna()
    g = g[g.tau_Qobs > 0]
    out["rho_all"] = stats.spearmanr(g.retention_obs, np.log(g.tau_Qobs)).statistic
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="derived_data")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    got = compute(load(a.derived))
    if not a.quiet:
        print("=== Text S2: memory governs the duration of the forced response ===")
        print(f"catchments with signal in both series : {got['n_signal']}")
        print(f"  of those, snow-free (frac_snow<={SNOW_FREE}) : {got['n_snow_free']}")
        print(f"snow-free rho(retention_obs, log tau_Qobs) = {got['rho_tau_obs']:+.3f}"
              f"  (n={got['n_tau_obs']})")
        print(f"snow-free rho(retention_obs, log tau_LZ)   = {got['rho_tau_LZ']:+.3f}"
              f"  (n={got['n_tau_LZ']})")
        print(f"  retention by tau_Qobs quartile: {got['quartiles_tau_obs']}")
        print(f"  retention by tau_LZ   quartile: {got['quartiles_tau_LZ']}")
        print(f"whole sample rho                           = {got['rho_all']:+.3f}")

    bad = []
    for key, (want, tol) in EXPECTED.items():
        if abs(got[key] - want) > tol:
            bad.append(f"  {key}: manuscript {want}, recomputed {got[key]:.3f} "
                       f"(tolerance {tol})")
    if bad:
        print("\nMISMATCH against the values quoted in Text S2:")
        print("\n".join(bad))
        sys.exit(1)
    if not a.quiet:
        print("\nOK: every value quoted in Text S2 reproduces from the derived products")


if __name__ == "__main__":
    main()

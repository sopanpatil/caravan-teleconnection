#!/usr/bin/env python
"""
verify_manuscript.py

Regression check on every quantitative claim the paper makes that can be
re-derived from the archived derived products, plus every parameter it states
that must match a constant in this code.

The point is that the paper's numbers should not have to be taken on trust.
Each claim below records the section it appears in, the value printed there,
and the computation that reproduces it. A mismatch is a failure, not a warning.

Usage:
    python verify_manuscript.py                       # uses ./derived_data
    python verify_manuscript.py --derived <dir>
    python verify_manuscript.py --quiet               # exit code only

Scope. Three groups, and the boundaries are deliberate:

  A  Claims re-derived from derived_data/. Checked here.
  B  Parameters the Methods states, checked against this repository's own
     constants, so prose and code cannot drift apart.
  C  Claims that are NOT machine-checkable from the archive. Listed by
     --list-unchecked with the reason for each, so the gap is explicit rather
     than silent. Two other scripts cover the Supporting Information:
     temperature_control.py (Text S1) and verify_text_s2.py
     (Text S2). Supporting Information Tables S1 to S3 are rendered directly
     from physiographic_*.csv, so they are checked by construction.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

CHECKS: list[tuple] = []

SIG_MIN = 0.2   # the peak |r| gate; must match response_timing.SIG_MIN


def _timing_pair(m):
    """Catchments where the timing properties are defined on BOTH series.

    The timing metrics are only meaningful where a teleconnection signal is
    present, so the comparison is gated on peak |r| in each series, not merely
    on both values being finite. Omitting the gate shifts the correlations by
    about 0.01 and is the commonest way to fail to reproduce them.
    """
    return (m.reg_lag_obs.notna() & m.reg_lag.notna()
            & (m.sig > SIG_MIN) & (m.sig_obs > SIG_MIN))


def claim(section, text, expected, tol):
    """Register a claim: @claim(...) decorates the function computing it."""
    def deco(fn):
        CHECKS.append((section, text, expected, tol, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
# Group A: re-derived from the archived products
# --------------------------------------------------------------------------

@claim("2.2", "3,201 catchments calibrated", 3201, 0)
def _(d): return len(d["cal"])

@claim("2.2", "2,286 pass both calibration and validation KGE > 0.5", 2286, 0)
def _(d):
    c = d["cal"]
    return int(((c.calibration_kge > 0.5) & (c.validation_kge > 0.5)).sum())

@claim("2.2", "screen removes 29 % of calibrated catchments", 29.0, 0.5)
def _(d):
    c = d["cal"]
    return 100 * (1 - ((c.calibration_kge > 0.5) & (c.validation_kge > 0.5)).sum() / len(c))

@claim("2.2", "2,135 catchments in the analysis set", 2135, 0)
def _(d): return int(d["cal"].include_in_analysis.astype(bool).sum())

@claim("2.2", "16 countries", 16, 0)
def _(d):
    # The raw country label splits Great Britain into England, Scotland, Wales
    # and "Great Britain"; physiographic_synthesis.py consolidates them, so the
    # analysis has 16 countries where the table has 19 labels.
    return d["inc"].country.replace(
        {"England": "Great Britain", "Scotland": "Great Britain",
         "Wales": "Great Britain"}).nunique()

@claim("2.2", "median catchment area 232 km2", 232, 1)
def _(d): return d["nest"].area.median()

@claim("2.2", "areas span under 2 to about 1.6e5 km2", 1.587e5, 2e3)
def _(d): return d["nest"].area.max()

@claim("2.3", "catchment areas sum to about 6.7e6 km2", 6.7e6, 5e4)
def _(d): return d["nest"].area.sum()

@claim("2.3", "1,364 spatially independent catchments", 1364, 0)
def _(d): return int(d["nest"].independent.astype(bool).sum())

@claim("2.3", "independent subset sums to 0.65e6 km2", 0.65e6, 1e4)
def _(d): return d["nest"].loc[d["nest"].independent.astype(bool), "area"].sum()

@claim("4.1", "median 67 winters per catchment", 67, 0)
def _(d): return d["s1"].n_winters.median()

# The record spans of Section 2.2. The Danish forcing starts late and the other
# three sources reach back to the ERA5-Land back-extension, so the sentence
# quotes a Danish span and a range for the rest; both are checked here.
_BACK = ["camelsgb", "lamah", "grdc"]   # the British, Alpine and GRDC sources

@claim("2.2", "Danish record opens in winter year 1987", 1987, 0)
def _(d): return d["join"].loc[d["join"].source == "camelsdk", "winter_year"].min()

@claim("2.2", "Danish record closes in 2020", 2020, 0)
def _(d): return d["join"].loc[d["join"].source == "camelsdk", "winter_year"].max()

@claim("2.2", "34 winters for Denmark", 34, 0)
def _(d): return d["s1"].loc[d["s1"].source == "camelsdk", "n_winters"].median()

@claim("2.2", "earliest winter of the other three sources is 1955/56", 1956, 0)
def _(d): return d["join"].loc[d["join"].source.isin(_BACK), "winter_year"].min()

@claim("2.2", "latest of their three start years is 1956/57", 1957, 0)
def _(d):
    j = d["join"][d["join"].source.isin(_BACK)]
    return j.groupby("source").winter_year.min().max()

@claim("2.2", "those three run to 2023", 2023, 0)
def _(d): return d["join"].loc[d["join"].source.isin(_BACK), "winter_year"].max()

@claim("2.2", "67 to 68 winters for those three: the shortest", 67, 0)
def _(d):
    j = d["join"][d["join"].source.isin(_BACK)]
    return j.groupby(["source", "gauge_id"]).size().groupby("source").median().min()

@claim("2.2", "67 to 68 winters for those three: the longest", 68, 0)
def _(d):
    j = d["join"][d["join"].source.isin(_BACK)]
    return j.groupby(["source", "gauge_id"]).size().groupby("source").median().max()

@claim("4.1", "modes explain a median 33 % of DJF precipitation variance", 33.0, 0.5)
def _(d): return 100 * d["s1"].adj_r2.median()

@claim("4.1", "interquartile range 14 to 46 %", 14.0, 0.5)
def _(d): return 100 * d["s1"].adj_r2.quantile(0.25)

@claim("4.1", "interquartile range 14 to 46 %", 46.0, 0.5)
def _(d): return 100 * d["s1"].adj_r2.quantile(0.75)

@claim("4.1", "84 % have at least one significant mode", 84.0, 0.5)
def _(d):
    s = d["s1"][["sig_NAO", "sig_EA", "sig_EAWR", "sig_SCA"]].astype(bool)
    return 100 * s.any(axis=1).mean()

@claim("4.1", "EAWR significant at 50 % of catchments", 50.0, 0.5)
def _(d): return 100 * d["s1"].sig_EAWR.astype(bool).mean()

@claim("4.1", "EA significant at 49 % of catchments", 49.0, 0.5)
def _(d): return 100 * d["s1"].sig_EA.astype(bool).mean()

@claim("4.1", "NAO significant at 34 % of catchments", 34.0, 0.5)
def _(d): return 100 * d["s1"].sig_NAO.astype(bool).mean()

@claim("4.1", "SCA significant at under 5 % of catchments", 4.5, 0.5)
def _(d): return 100 * d["s1"].sig_SCA.astype(bool).mean()

@claim("4.1", "median beta_EAWR -0.24 mm/day per unit index", -0.24, 0.005)
def _(d): return d["s1"].beta_EAWR.median()

@claim("4.1", "median beta_EA +0.13", 0.13, 0.005)
def _(d): return d["s1"].beta_EA.median()

@claim("4.1", "median beta_NAO +0.21", 0.21, 0.005)
def _(d): return d["s1"].beta_NAO.median()

@claim("4.2.1", "median transfer coefficient 0.72", 0.72, 0.005)
def _(d): return d["s2"].gain_Qsim.median()

@claim("4.2.1", "interquartile range 0.30 to 0.97", 0.30, 0.005)
def _(d): return d["s2"].gain_Qsim.quantile(0.25)

@claim("4.2.1", "interquartile range 0.30 to 0.97", 0.97, 0.005)
def _(d): return d["s2"].gain_Qsim.quantile(0.75)

@claim("4.2.1", "gain exceeds 1 at 19 % of catchments", 19.0, 0.5)
def _(d): return 100 * (d["s2"].gain_Qsim > 1).mean()

@claim("4.2.1", "gain below 0 at 5 % of catchments", 5.0, 0.5)
def _(d): return 100 * (d["s2"].gain_Qsim < 0).mean()

@claim("4.2.1", "0.89 in snow-free catchments", 0.89, 0.005)
def _(d): return d["s2"].loc[d["s2"].sp_active_frac < 0.30, "gain_Qsim"].median()

@claim("4.2.1", "0.17 in snow-active catchments", 0.17, 0.005)
def _(d): return d["s2"].loc[d["s2"].sp_active_frac >= 0.30, "gain_Qsim"].median()

@claim("4.2.2", "tau_UZ about 5 days", 5.0, 0.1)
def _(d): return d["s2"].tau_UZ.median()

@claim("4.2.2", "tau_Qsim about 9 days", 9.0, 0.15)
def _(d): return d["s2"].tau_Qsim.median()

@claim("4.2.2", "tau_SM about 53 days", 53.0, 0.5)
def _(d): return d["s2"].tau_SM.median()

@claim("4.2.2", "tau_LZ about 32 days", 32.0, 0.5)
def _(d): return d["s2"].tau_LZ.median()

@claim("4.2.2", "721 snow-active catchments", 721, 0)
def _(d): return int((d["s2"].sp_active_frac >= 0.30).sum())

@claim("4.2.2", "706 of them have a resolvable snowpack memory", 706, 0)
def _(d):
    s = d["s2"]
    return int(s.loc[s.sp_active_frac >= 0.30, "tau_SP"].notna().sum())

@claim("4.2.2", "tau_SP about 66 days", 66.0, 0.5)
def _(d):
    s = d["s2"]
    return s.loc[s.sp_active_frac >= 0.30, "tau_SP"].median()

@claim("4.2.2", "tau_LZ reaches 1,101 days at the 99th percentile", 1101, 3)
def _(d): return d["s2"].tau_LZ.quantile(0.99)

@claim("4.2.2", "tau_LZ resolvable at 2,000 catchments", 2000, 0)
def _(d): return int(d["s2"].tau_LZ.notna().sum())

@claim("4.2.2", "258 catchments (12.9 %) exceed 180 days", 258, 0)
def _(d): return int((d["s2"].tau_LZ > 180).sum())

@claim("4.2.2", "192 (9.6 %) exceed one year", 192, 0)
def _(d): return int((d["s2"].tau_LZ > 365).sum())

@claim("4.2.3", "response strength against response lag, Spearman -0.69", -0.69, 0.006)
def _(d):
    m = d["m"]
    ok = m.gain_Qsim.notna() & m.reg_lag.notna()
    return spearmanr(m.gain_Qsim[ok], m.reg_lag[ok]).statistic

@claim("4.2.3", "memory against response lag, Spearman +0.25", 0.25, 0.006)
def _(d):
    m = d["m"]
    ok = m.tau_Qsim.notna() & m.reg_lag.notna()
    return spearmanr(m.tau_Qsim[ok], m.reg_lag[ok]).statistic

@claim("4.2.3", "memory against response strength, Spearman -0.35", -0.35, 0.006)
def _(d):
    m = d["m"]
    ok = m.tau_Qsim.notna() & m.gain_Qsim.notna()
    return spearmanr(m.tau_Qsim[ok], m.gain_Qsim[ok]).statistic

@claim("4.3", "observed memory resolvable at all 2,135 catchments", 2135, 0)
def _(d): return int(d["obs"].tau_Qobs.notna().sum())

@claim("4.3", "median observed tau 7.4 days", 7.4, 0.05)
def _(d): return d["obs"].tau_Qobs.median()

@claim("4.3", "interquartile range 4.0 to 12.9 days", 4.0, 0.05)
def _(d): return d["obs"].tau_Qobs.quantile(0.25)

@claim("4.3", "interquartile range 4.0 to 12.9 days", 12.9, 0.05)
def _(d): return d["obs"].tau_Qobs.quantile(0.75)

@claim("4.3", "observed against gap-masked simulated memory, Spearman +0.88", 0.88, 0.006)
def _(d):
    o = d["obs"]
    ok = o.tau_Qobs.notna() & o.tau_Qsim_m.notna()
    return spearmanr(o.tau_Qobs[ok], o.tau_Qsim_m[ok]).statistic

@claim("4.3", "bounded lag-1 proxy, Spearman +0.90", 0.90, 0.006)
def _(d):
    o = d["obs"]
    ok = o.ac1_Qobs.notna() & o.ac1_Qsim_m.notna()
    return spearmanr(o.ac1_Qobs[ok], o.ac1_Qsim_m[ok]).statistic

@claim("4.3", "model over-persistent by a median factor of 1.27", 1.27, 0.006)
def _(d): return (d["obs"].tau_Qsim_m / d["obs"].tau_Qobs).median()

@claim("4.3", "late-response fraction, observed against simulated, Spearman +0.71", 0.71, 0.006)
def _(d):
    m = d["m"]
    ok = _timing_pair(m)
    return spearmanr(m.late_frac_obs[ok], m.late_frac[ok]).statistic

@claim("4.3", "response lag, observed against simulated, Spearman +0.68", 0.68, 0.006)
def _(d):
    m = d["m"]
    ok = _timing_pair(m)
    return spearmanr(m.reg_lag_obs[ok], m.reg_lag[ok]).statistic

@claim("4.3", "observed median response lag 3.0 months", 3.0, 0.05)
def _(d): return d["obs"].reg_lag_obs.median()

@claim("4.3", "observed median late fraction 0.28", 0.28, 0.005)
def _(d): return d["obs"].late_frac_obs.median()

@claim("4.3", "simulated median response lag 2.6 months", 2.6, 0.05)
def _(d): return d["s2c"].reg_lag.median()

@claim("4.3", "simulated median late fraction 0.22", 0.22, 0.005)
def _(d): return d["s2c"].late_frac.median()

@claim("4.3", "observed late fraction against snow-active fraction, Spearman +0.59", 0.59, 0.006)
def _(d):
    m = d["m"]
    ok = m.late_frac_obs.notna() & m.sp_active_frac.notna()
    return spearmanr(m.late_frac_obs[ok], m.sp_active_frac[ok]).statistic

@claim("4.3", "snow-free observed late fraction 0.17", 0.17, 0.006)
def _(d):
    m = d["m"]
    return m.loc[m.sp_active_frac < 0.30, "late_frac_obs"].median()

@claim("4.3", "snow-free observed response lag 2.5 months", 2.5, 0.06)
def _(d):
    m = d["m"]
    return m.loc[m.sp_active_frac < 0.30, "reg_lag_obs"].median()

@claim("4.3", "snow-active observed late fraction 0.53", 0.53, 0.006)
def _(d):
    m = d["m"]
    return m.loc[m.sp_active_frac >= 0.30, "late_frac_obs"].median()

@claim("4.3", "snow-active observed response lag 4.0 months", 4.0, 0.06)
def _(d):
    m = d["m"]
    return m.loc[m.sp_active_frac >= 0.30, "reg_lag_obs"].median()

@claim("4.3", "response strength estimable on both series at 2,019 catchments", 2019, 0)
def _(d):
    g = d["gain"]
    return int((g.gain_obs.notna() & g.gain_sim.notna()).sum())

@claim("4.3", "observed against simulated transfer coefficient, Spearman +0.85", 0.85, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return spearmanr(g.gain_obs[ok], g.gain_sim[ok]).statistic

@claim("4.3", "Pearson +0.79", 0.79, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return pearsonr(g.gain_obs[ok], g.gain_sim[ok]).statistic

@claim("4.3", "paired median difference -0.021", -0.021, 0.0015)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return (g.gain_sim[ok] - g.gain_obs[ok]).median()

@claim("4.3", "observed 5th to 95th percentile 0.04 to 1.67", 0.04, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return g.gain_obs[ok].quantile(0.05)

@claim("4.3", "observed 5th to 95th percentile 0.04 to 1.67", 1.67, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return g.gain_obs[ok].quantile(0.95)

@claim("4.3", "simulated 5th to 95th percentile 0.00 to 1.12", 0.00, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return g.gain_sim[ok].quantile(0.05)

@claim("4.3", "simulated 5th to 95th percentile 0.00 to 1.12", 1.12, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return g.gain_sim[ok].quantile(0.95)

@claim("4.3", "tau_LZ against 1/K2, Spearman +0.87", 0.87, 0.006)
def _(d):
    m = d["s2"].merge(d["cal"][["gauge_id", "K2"]], on="gauge_id")
    ok = m.tau_LZ.notna() & m.K2.notna()
    return spearmanr(m.tau_LZ[ok], 1.0 / m.K2[ok]).statistic

@claim("4.4", "marginal R2 0.18 for observed log tau", 0.18, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "marginal_R2"]

@claim("4.4", "marginal R2 0.29 for the bounded observed proxy", 0.29, 0.006)
def _(d): return d["sum"].set_index("response").loc["logit_ac1_obs", "marginal_R2"]

@claim("4.4", "attributes add 0.02 over country for observed response lag", 0.02, 0.006)
def _(d): return d["sum"].set_index("response").loc["reg_lag_obs", "predictor_partial_R2_within"]

@claim("4.4", "0.04 for the observed late fraction", 0.04, 0.006)
def _(d): return d["sum"].set_index("response").loc["logit_late_obs", "predictor_partial_R2_within"]

@claim("4.4", "0.07 for observed log tau", 0.07, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "predictor_partial_R2_within"]

@claim("4.4", "0.11 for its simulated counterpart", 0.11, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau", "predictor_partial_R2_within"]

@claim("4.4", "ICC 0.28 for observed response strength", 0.28, 0.006)
def _(d): return d["sum"].set_index("response").loc["gain_Qobs", "icc"]

@claim("4.4", "timing regressions use 2,107 catchments", 2107, 0)
def _(d): return int(d["sum"].set_index("response").loc["reg_lag", "n"])

@claim("4.5", "random 10-fold R2 +0.17 for observed memory", 0.17, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "kfold_r2_global"]

@claim("4.5", "+0.26 for observed timing", 0.26, 0.006)
def _(d): return d["sum"].set_index("response").loc["reg_lag_obs", "kfold_r2_global"]

@claim("4.5", "+0.34 for observed response strength", 0.34, 0.006)
def _(d): return d["sum"].set_index("response").loc["gain_Qobs", "kfold_r2_global"]

@claim("4.5", "LOCO against sample mean +0.18 for bounded observed memory", 0.18, 0.006)
def _(d): return d["sum"].set_index("response").loc["logit_ac1_obs", "loco_r2_global"]

@claim("4.5", "+0.13 for its simulated counterpart", 0.13, 0.006)
def _(d): return d["sum"].set_index("response").loc["logit_ac1", "loco_r2_global"]

@claim("4.5", "+0.03 for observed timing", 0.03, 0.006)
def _(d): return d["sum"].set_index("response").loc["reg_lag_obs", "loco_r2_global"]

@claim("4.5", "-0.02 for observed log tau", -0.02, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "loco_r2_global"]

@claim("4.5", "LOCO against local mean -0.23 for observed memory", -0.23, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "loco_r2_local_pooled"]

@claim("4.5", "-0.67 for observed timing", -0.67, 0.006)
def _(d): return d["sum"].set_index("response").loc["reg_lag_obs", "loco_r2_local_pooled"]

@claim("4.5", "-1.27 for observed response strength", -1.27, 0.006)
def _(d): return d["sum"].set_index("response").loc["gain_Qobs", "loco_r2_local_pooled"]

@claim("4.5", "below the local mean in 12 of the 15 countries where defined", 12, 0)
def _(d):
    b = d["by"]
    return int((b.r2_local < 0).sum())

@claim("4.5", "15 countries where the local score is defined", 15, 0)
def _(d): return int(d["by"].r2_local.notna().sum())

@claim("4.5", "median -0.185 across the 14 countries with n >= 10", -0.185, 0.0006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "loco_r2_local_median_big"]

@claim("4.5", "de-biasing lifts that median to -0.052", -0.052, 0.0006)
def _(d):
    return d["sum"].set_index("response").loc["log_tau_obs", "loco_r2_local_debiased_median_big"]

@claim("4.5", "count below own mean falls from 11 to 8 of 14", 11, 0)
def _(d): return int(d["sum"].set_index("response").loc["log_tau_obs", "loco_n_big_negative_local"])

@claim("4.5", "count below own mean falls from 11 to 8 of 14", 8, 0)
def _(d): return int(d["sum"].set_index("response").loc["log_tau_obs", "loco_n_big_negative_debiased"])

@claim("4.5", "14 countries retain at least 10 catchments", 14, 0)
def _(d): return int(d["sum"].set_index("response").loc["log_tau_obs", "loco_n_countries_big"])

@claim("4.5", "median within-country Spearman +0.06", 0.059, 0.0015)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "loco_rho_within_median_big"]

@claim("4.5", "five of 14 order no better than at random", 5, 0)
def _(d): return int(d["sum"].set_index("response").loc["log_tau_obs", "loco_n_big_negative_rho"])

@claim("4.5", "pooled de-biased -0.11 for observed response strength", -0.11, 0.006)
def _(d): return d["sum"].set_index("response").loc["gain_Qobs", "loco_r2_local_debiased_pooled"]

@claim("4.5", "-0.11 for observed timing", -0.11, 0.006)
def _(d): return d["sum"].set_index("response").loc["reg_lag_obs", "loco_r2_local_debiased_pooled"]

@claim("4.5", "-0.05 for observed memory", -0.05, 0.006)
def _(d): return d["sum"].set_index("response").loc["log_tau_obs", "loco_r2_local_debiased_pooled"]

@claim("4.6", "independent subset local reference -0.24 for memory", -0.24, 0.006)
def _(d): return d["sumi"].set_index("response").loc["log_tau_obs", "loco_r2_local_pooled"]

@claim("4.6", "-1.04 for response strength", -1.04, 0.006)
def _(d): return d["sumi"].set_index("response").loc["gain_Qobs", "loco_r2_local_pooled"]

@claim("4.6", "-0.85 for timing", -0.85, 0.006)
def _(d): return d["sumi"].set_index("response").loc["reg_lag_obs", "loco_r2_local_pooled"]

@claim("4.6", "independent median local R2 -0.221 before de-biasing", -0.221, 0.0006)
def _(d): return d["sumi"].set_index("response").loc["log_tau_obs", "loco_r2_local_median_big"]

@claim("4.6", "-0.084 after", -0.084, 0.0006)
def _(d):
    return d["sumi"].set_index("response").loc["log_tau_obs", "loco_r2_local_debiased_median_big"]

@claim("SI Text S3", "639 same-stem gauge pairs", 639, 0)
def _(d): return len(d["pairs"])

@claim("SI Text S3", "independent median within-country Spearman +0.14", 0.14, 0.006)
def _(d): return d["sumi"].set_index("response").loc["log_tau_obs", "loco_rho_within_median_big"]

@claim("Abstract", "Spearman +0.85 for response strength", 0.85, 0.006)
def _(d):
    g = d["gain"]
    ok = g.gain_obs.notna() & g.gain_sim.notna()
    return spearmanr(g.gain_obs[ok], g.gain_sim[ok]).statistic

@claim("Abstract", "Spearman +0.88 for memory", 0.88, 0.006)
def _(d):
    o = d["obs"]
    ok = o.tau_Qobs.notna() & o.tau_Qsim_m.notna()
    return spearmanr(o.tau_Qobs[ok], o.tau_Qsim_m[ok]).statistic


# --------------------------------------------------------------------------
# Group B: Methods parameters, checked against this repository's constants
# --------------------------------------------------------------------------

def _const(module_file: str, name: str):
    """Read a module-level constant from source, without importing.

    Importing would pull in hbv_model, which is an external dependency this
    check should not need. Parsing keeps the script runnable in a bare clone.
    """
    import ast

    class _Environ:                       # os.environ.get(KEY, default) -> default
        @staticmethod
        def get(_key, default=None):
            return default

    class _Os:
        environ = _Environ()

    safe = {"os": _Os(), "int": int, "float": float, "str": str,
            "list": list, "range": range, "tuple": tuple, "set": set}

    here = os.path.dirname(os.path.abspath(__file__))
    tree = ast.parse(open(os.path.join(here, module_file), encoding="utf-8").read())
    def _eval(value):
        try:
            return ast.literal_eval(value)
        except ValueError:
            expr = ast.Expression(body=value)
            code = compile(ast.fix_missing_locations(expr), "<constant>", "eval")
            return eval(code, {"__builtins__": {}}, safe)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Tuple assignment, e.g. CAL_START, CAL_END = "...", "..."
            for tgt in node.targets:
                if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for elt, val in zip(tgt.elts, node.value.elts):
                        if isinstance(elt, ast.Name) and elt.id == name:
                            return _eval(val)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        # A constant defined as an expression, e.g. an env-var
                        # override with a literal default, or list(range(...)).
                        expr = ast.Expression(body=node.value)
                        code = compile(ast.fix_missing_locations(expr),
                                       "<constant>", "eval")
                        return eval(code, {"__builtins__": {}}, safe)
    raise KeyError(f"{name} not found in {module_file}")


def group_b() -> list[tuple[str, str, object, object]]:
    """Methods parameters, each checked against the constant that implements it."""
    spec = [
        ("3.2", "SCE-UA with seven complexes",
         7, "calibrate_catchment.py", "N_COMPLEXES"),
        ("3.2", "calibration window opens in winter year 1983",
         "1983-01-01", "calibrate_catchment.py", "CAL_START"),
        ("3.2", "validation window closes in 2020",
         "2020-12-31", "calibrate_catchment.py", "VAL_END"),
        ("3.2", "first five years discarded as spin-up",
         5, "generate_states.py", "SPINUP_YEARS"),
        ("2.2", "glacier cover above 5 % excluded",
         5.0, "screen_analysis_sample.py", "GLACIER_MAX"),
        ("3.1", "seasonal mean needs 80 of the 90 winter days",
         80, "build_seasonal_table.py", "DEFAULT_MIN_DAYS"),
        ("3.3.1", "gain estimated where at least 20 winters are available",
         20, "fit_precipitation_signal.py", "MIN_WINTERS"),
        ("3.3.2", "e-folding searched within a five-year horizon",
         1825, "response_strength_and_memory.py", "KMAX"),
        ("3.3.2", "at least 3,000 valid days for a memory estimate",
         3000, "response_properties_observed.py", "MIN_DAYS"),
        ("3.3.2", "at least 365 valid day-pairs before a lag is admitted",
         365, "response_properties_observed.py", "MIN_PAIRS"),
        ("3.3.3", "at least 20 valid days before a month contributes",
         20, "response_properties_observed.py", "MIN_MONTH_DAYS"),
        ("3.3.3", "lag profile computed where peak |r| > 0.2",
         0.2, "response_timing.py", "SIG_MIN"),
        ("3.3.3", "lags 0 to 11 months after the December",
         list(range(0, 12)), "response_timing.py", "LAGS"),
        ("3.4", "wild cluster bootstrap with 1,999 replicates",
         1999, "physiographic_synthesis.py", "NBOOT"),
        ("3.4", "summaries restricted to countries with at least 10 catchments",
         10, "physiographic_synthesis.py", "MIN_COUNTRY_N"),
        ("4.2.1", "snow-active means a snowpack anomaly in at least 30 % of winters",
         0.3, "temperature_control.py", "SP_ACTIVE"),
        ("SI Text S2", "snow-free means HydroATLAS snow fraction at most 0.05",
         0.05, "verify_text_s2.py", "SNOW_FREE"),
    ]
    return [(sec, text, stated, _const(f, n)) for sec, text, stated, f, n in spec]


UNCHECKED = [
    ("2.2", "Table 1 median KGE by source",
     "derivable, but reported per Caravan source rather than per country; "
     "check with a groupby on calibrated_parameters_ALL_refined.csv"),
    ("2.2", "observed discharge median 37 years, minimum 10",
     "needs the daily observed series, which the archive does not carry"),
    ("3.3.2", "observed records median 63 % coverage",
     "same: needs the daily series"),
    ("4.2.1", "standardised store gains 0.42, 0.36, 0.26, 0.46, -0.08",
     "computed during plotting from store anomaly standard deviations that are "
     "not themselves archived"),
    ("4.2.1", "deep versus shallow water table 0.90 against 0.88",
     "derivable, but the split is a median on gwt_depth and the paper does not "
     "state the threshold; see plot scripts"),
    ("4.2.3", "endpoint medians 1.7 and 4.5 months, 0.07 and 0.61, +0.51 and +0.13",
     "grouped by named country sets (Britain and Ireland; the Alps) that the "
     "archive does not carry as a field"),
    ("4.1", "latitude-band means -0.52, +0.41 and their significance shares",
     "reproduced by figures/fig1_nao_precipitation_sensitivity.py, which prints them"),
    ("4.4", "per-predictor coefficients, p and q",
     "Supporting Information Table S2 is rendered from physiographic_coeffs_*.csv"),
    ("4.5", "per-country skill",
     "Supporting Information Table S3 is rendered from "
     "physiographic_by_country_full.csv"),
    ("SI Text S1", "temperature-control values",
     "checked by temperature_control.py"),
    ("SI Text S2", "retention values",
     "checked by verify_text_s2.py"),
    ("Figures", "all five",
     "regenerate with the figures/fig*.py scripts and compare; figures 1, 2, 4 and 5 "
     "rebuild from derived_data/ alone, figure 3 needs the daily state series"),
]


def load(derived: str) -> dict:
    p = lambda f: os.path.join(derived, f)
    d = {}
    d["cal"] = pd.read_csv(p("calibrated_parameters_ALL_refined.csv"))
    d["inc"] = d["cal"][d["cal"].include_in_analysis.astype(bool)]
    d["s1"] = pd.read_parquet(p("precipitation_signal_DJF.parquet"))
    # Only the three columns the record-span claims need: the full join is 25 MB.
    d["join"] = pd.read_parquet(p("seasonal_join_DJF.parquet"),
                                columns=["winter_year", "gauge_id", "source"])
    d["s2"] = pd.read_parquet(p("response_strength_memory_DJF.parquet"))
    d["s2c"] = pd.read_parquet(p("response_timing_DJF.parquet"))
    d["obs"] = pd.read_parquet(p("response_observed_DJF.parquet"))
    d["gain"] = pd.read_parquet(p("gain_temperature_control.parquet"))
    d["nest"] = pd.read_csv(p("nesting_flags.csv"))
    d["pairs"] = pd.read_csv(p("nesting_stem_pairs.csv"))
    d["sum"] = pd.read_csv(p("physiographic_summary_full.csv"))
    d["sumi"] = pd.read_csv(p("physiographic_summary_independent.csv"))
    by = pd.read_csv(p("physiographic_by_country_full.csv"))
    d["by"] = by[by.response == "log_tau_obs"]
    m = d["s2"].merge(d["s2c"].drop(columns=[c for c in d["s2c"].columns
                                             if c in d["s2"].columns and c != "gauge_id"]),
                      on="gauge_id")
    m = m.merge(d["obs"].drop(columns=["source"]), on="gauge_id")
    d["m"] = m
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", default="derived_data")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list-unchecked", action="store_true",
                    help="print the claims this script does not cover, and why")
    a = ap.parse_args()

    if a.list_unchecked:
        print("Claims not checked here, with the reason:\n")
        for sec, what, why in UNCHECKED:
            print(f"  [{sec}] {what}\n      {why}\n")
        return 0

    d = load(a.derived)
    fails = []

    if not a.quiet:
        print(f"=== Group A: {len(CHECKS)} claims re-derived from {a.derived}/ ===")
    for sec, text, exp, tol, fn in CHECKS:
        got = float(fn(d))
        ok = abs(got - float(exp)) <= tol
        if not ok:
            fails.append(f"[{sec}] {text}: paper {exp}, archive {got:.4f}")
        if not a.quiet:
            print(f"  {'ok  ' if ok else 'FAIL'} [{sec:9s}] {text}"
                  f"{'' if ok else f'  -> archive gives {got:.4f}'}")

    b = group_b()
    if not a.quiet:
        print(f"\n=== Group B: {len(b)} Methods parameters against code constants ===")
    for sec, text, stated, actual in b:
        ok = stated == actual
        if not ok:
            fails.append(f"[{sec}] {text}: paper {stated!r}, code {actual!r}")
        if not a.quiet:
            print(f"  {'ok  ' if ok else 'FAIL'} [{sec:9s}] {text}"
                  f"{'' if ok else f'  -> code has {actual!r}'}")

    if not a.quiet:
        print(f"\n=== Group C: {len(UNCHECKED)} claims not covered "
              f"(--list-unchecked for the reasons) ===")

    if fails:
        print(f"\nFAILED: {len(fails)} claim(s) do not match the archive:")
        for f in fails:
            print("  " + f)
        return 1
    if not a.quiet:
        print(f"\nOK: {len(CHECKS)} archived claims and {len(b)} stated parameters "
              f"all reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
physiographic_synthesis.py

The physiographic synthesis and the cross-national transfer
test, for ALL THREE filter properties, with the inference and the skill reporting
put on a defensible footing. This replaces an earlier trio of synthesis scripts,
not included in this release, which between them covered only memory and timing,
reported p-values that 16 clusters cannot support, and scored transfer against a
baseline that flatters it.

Three filter properties, each a different thing the catchment does to the winter
teleconnection signal:

  AMPLITUDE (gain)  the share of the fitted winter precipitation anomaly that reaches
                    the gauge in the same winter. Teleconnection-derived: it is the
                    regression slope of the DJF flow anomaly on the fitted
                    forcing. Computed here for BOTH simulated and observed flow.
  MEMORY (tau, ac1) how long the catchment holds an anomaly. Note this is an INTRINSIC
                    property: it is read off the flow series itself and no
                    teleconnection index enters it.
  TIMING (lag)      when the catchment releases the signal over the water year.
                    Teleconnection-derived, from the timing/observed lag profile.

Each property is fitted on simulated and, where available, observed flow, so that a
model-derived result can be checked against the data that constrained it.

Two corrections to the earlier reporting:

  WILD CLUSTER BOOTSTRAP. The cluster-robust sandwich is badly downward-biased at
    G=16 clusters, so the earlier p-values were anti-conservative. Significance is now
    from a restricted wild cluster bootstrap (Cameron-Gelbach-Miller) with Rademacher
    weights: impose the null on the coefficient being tested, resample residuals by
    country, and refer the observed cluster-robust t to its bootstrap distribution.
  LOCAL-BASELINE TRANSFER SKILL. Out-of-sample R2 against the GLOBAL mean gives a model
    credit merely for placing each country at roughly the right level. The question
    that matters for prediction in an unseen country is whether the fit beats THAT
    COUNTRY'S OWN MEAN, so both are reported: r2_global (comparable with the earlier
    numbers) and r2_local (the honest within-country skill), pooled, per country, and
    summarised as a median and a catchment-weighted mean over countries.

    python physiographic_synthesis.py --strength-memory <response_strength_memory_DJF.parquet> \
        --timing <response_timing_DJF.parquet> --observed <response_observed_DJF.parquet> \
        --join <seasonal_join_DJF.parquet> --signal <precipitation_signal_DJF.parquet> \
        --attrs <attributes.parquet> --refined <..._refined.csv> \
        --nesting <nesting_flags.csv> --outdir <caravan_derived/> [--independent-only]
    python physiographic_synthesis.py --selftest
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["aridity", "frac_snow", "slope_l", "clay_pc", "gwt_l", "karst_pc"]
KFOLDS = 10
SEED = 0
MIN_COUNTRY_N = 10   # countries smaller than this give an unstable (or undefined) local
                     # R2; summary statistics over countries are restricted to these
SIG_MIN = 0.2        # peak |r| gate on a lag profile, as in response_timing
NBOOT = 1999
MIN_WINTERS_GAIN = 20
GB = {"England", "Scotland", "Wales", "Great Britain"}
BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
        "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}

# (response column, family, label, whether it is teleconnection-derived)
PROPERTIES = [
    ("gain_Qsim",      "amplitude", "amplitude: winter transfer coeff (simulated)", True),
    ("gain_Qobs",      "amplitude", "amplitude: winter transfer coeff (OBSERVED)",  True),
    ("log_tau",        "memory",    "memory: log tau (simulated)",                  False),
    ("logit_ac1",      "memory",    "memory: logit ac1 (simulated)",                False),
    ("log_tau_obs",    "memory",    "memory: log tau (OBSERVED)",                   False),
    ("logit_ac1_obs",  "memory",    "memory: logit ac1 (OBSERVED)",                 False),
    ("reg_lag",        "timing",    "timing: response lag (simulated)",             True),
    ("logit_late",     "timing",    "timing: logit late fraction (simulated)",      True),
    ("reg_lag_obs",    "timing",    "timing: response lag (OBSERVED)",              True),
    ("logit_late_obs", "timing",    "timing: logit late fraction (OBSERVED)",       True),
]


def bh_fdr(p):
    """Benjamini-Hochberg q-values. Family = the six predictors within one response."""
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate(
        (p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def logit(x, lo=1e-4):
    x = np.clip(np.asarray(x, dtype=float), lo, 1 - lo)
    return np.log(x / (1 - x))


# ----------------------------------------------------------------------- assembly

def observed_gain(join: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Gain of the OBSERVED DJF flow anomaly on the fitted winter forcing."""
    b = signal.set_index("gauge_id")
    rows = []
    for gid, g in join.groupby("gauge_id", sort=False):
        if gid not in b.index:
            continue
        bb = b.loc[gid]
        phat = sum(bb[BETA[c]] * g[c] for c in BETA)
        d = pd.DataFrame({"p": phat, "y": g["flow_obs_DJF_anom"]}).dropna()
        if len(d) < MIN_WINTERS_GAIN or d["p"].var() == 0:
            rows.append({"gauge_id": gid, "gain_Qobs": np.nan, "n_winters_gain_obs": len(d)})
        else:
            rows.append({"gauge_id": gid,
                         "gain_Qobs": float(np.polyfit(d["p"], d["y"], 1)[0]),
                         "n_winters_gain_obs": len(d)})
    return pd.DataFrame(rows)


def prepare(strength_memory, timing, observed, gain_obs, attrs, refined, nesting):
    d = strength_memory.merge(attrs, on="gauge_id", how="left")
    country = refined.set_index("gauge_id")["country"]
    d["country"] = d.gauge_id.map(country)
    d["country"] = d["country"].where(~d["country"].isin(GB), "Great Britain")

    d = d.merge(timing[["gauge_id", "reg_lag", "late_frac", "sig"]], on="gauge_id", how="left")
    if observed is not None:
        keep = ["gauge_id", "tau_Qobs", "ac1_Qobs", "tau_Qsim_m", "ac1_Qsim_m",
                "reg_lag_obs", "late_frac_obs", "sig_obs"]
        d = d.merge(observed[[c for c in keep if c in observed.columns]],
                    on="gauge_id", how="left")
    if gain_obs is not None:
        d = d.merge(gain_obs, on="gauge_id", how="left")
    if nesting is not None:
        d = d.merge(nesting[["gauge_id", "independent", "n_gauges_inside"]],
                    on="gauge_id", how="left")

    d["slope_l"] = np.log1p(d["slope_deg"])
    d["gwt_l"] = np.log1p(d["gwt_depth"])
    d["log_tau"] = np.where(d.tau_Qsim > 0, np.log(d.tau_Qsim.where(d.tau_Qsim > 0)), np.nan)
    d["logit_ac1"] = logit(d["ac1_Qsim"])
    if "tau_Qobs" in d:
        d["log_tau_obs"] = np.where(d.tau_Qobs > 0, np.log(d.tau_Qobs.where(d.tau_Qobs > 0)), np.nan)
        d["logit_ac1_obs"] = logit(d["ac1_Qobs"])
    # timing is undefined where the modes barely control the catchment's precipitation
    d["logit_late"] = np.where(d["sig"] > SIG_MIN, logit(d["late_frac"]), np.nan)
    d["reg_lag"] = d["reg_lag"].where(d["sig"] > SIG_MIN)
    if "sig_obs" in d:
        d["logit_late_obs"] = np.where(d["sig_obs"] > SIG_MIN, logit(d["late_frac_obs"]), np.nan)
        d["reg_lag_obs"] = d["reg_lag_obs"].where(d["sig_obs"] > SIG_MIN)
    return d.dropna(subset=["country"] + PREDICTORS).reset_index(drop=True)


# ---------------------------------------------------------------------- inference

def ols_cluster(X, y, groups):
    """OLS with country-cluster-robust covariance (statsmodels' finite-sample scaling).
    X excludes the intercept, which is added here. Returns beta, se, t, R2."""
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    XtXi = np.linalg.pinv(Z.T @ Z)
    beta = XtXi @ Z.T @ y
    u = y - Z @ beta
    meat = np.zeros((k + 1, k + 1))
    uniq = pd.unique(groups)
    for g in uniq:
        m = groups == g
        Zg = Z[m]
        ug = u[m]
        s = Zg.T @ ug
        meat += np.outer(s, s)
    G = len(uniq)
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k - 1, 1))
    V = c * (XtXi @ meat @ XtXi)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    r2 = 1 - (u @ u) / (((y - y.mean()) ** 2).sum())
    return beta[1:], se[1:], t[1:], float(r2), u


def wild_cluster_bootstrap(X, y, groups, j, nboot=NBOOT, seed=SEED):
    """Restricted wild cluster bootstrap p-value for predictor j (Rademacher weights).

    Impose H0: beta_j = 0 by refitting without column j, then resample the restricted
    residuals with a +-1 sign flipped once per COUNTRY. The observed cluster-robust t
    is referred to the bootstrap distribution of t*, which is the right reference at
    G=16 where the asymptotic normal reference is not."""
    _, _, t_obs, _, _ = ols_cluster(X, y, groups)
    t0 = t_obs[j]
    if not np.isfinite(t0):
        return np.nan, np.nan
    Xr = np.delete(X, j, axis=1)
    n = X.shape[0]
    Zr = np.column_stack([np.ones(n), Xr])
    br = np.linalg.pinv(Zr.T @ Zr) @ Zr.T @ y
    fit_r = Zr @ br
    res_r = y - fit_r

    uniq, gidx = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed + 1000 * j)
    count = 0
    valid = 0
    for _ in range(nboot):
        w = rng.choice(np.array([-1.0, 1.0]), size=len(uniq))[gidx]
        ystar = fit_r + w * res_r
        _, _, tstar, _, _ = ols_cluster(X, ystar, groups)
        ts = tstar[j]
        if np.isfinite(ts):
            valid += 1
            count += abs(ts) >= abs(t0)
    p = (count + 1) / (valid + 1) if valid else np.nan
    return float(p), float(t0)


def mom_icc(resid, country):
    """Method-of-moments ICC: between-country variance / total."""
    df = pd.DataFrame({"r": np.asarray(resid), "c": np.asarray(country)})
    gm = df.groupby("c")["r"]
    means, sizes, var = gm.mean(), gm.size(), gm.var(ddof=1)
    w = sizes / sizes.sum()
    between = float((w * (means - (w * means).sum()) ** 2).sum())
    within = float(np.nansum((sizes - 1) * var) / max(sizes.sum() - len(sizes), 1))
    tot = between + within
    return between / tot if tot > 0 else np.nan


def variance_split(X, y, groups):
    """Decompose explained variance into predictor, country, and shared parts.

    Returns (marginal, country_partial, predictor_partial_within):
      marginal                   R2 of the six predictors alone
      country_partial            R2 gained by ADDING country indicators to them
      predictor_partial_within   R2 gained by ADDING the predictors to country
                                 indicators alone. This is the diagnostic that
                                 separates "the relationship does not transfer" from
                                 "there is no within-country signal to transfer": if it
                                 is near zero, physiography orders countries and says
                                 almost nothing about catchments inside one, so a
                                 within-country transfer test cannot succeed even in
                                 principle.
    """
    n = X.shape[0]
    tss = ((y - y.mean()) ** 2).sum()
    Z = np.column_stack([np.ones(n), X])
    D = pd.get_dummies(pd.Series(groups), drop_first=True).to_numpy(float)
    r_pred = 1 - _rss(Z, y) / tss
    r_ctry = 1 - _rss(np.column_stack([np.ones(n), D]), y) / tss
    r_full = 1 - _rss(np.column_stack([Z, D]), y) / tss
    return float(r_pred), float(r_full - r_pred), float(r_full - r_ctry)


def _rss(Z, y):
    b = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    u = y - Z @ b
    return float(u @ u)


# ------------------------------------------------------------------ cross-validation

def _fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Ztr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    Zte = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    beta, *_ = np.linalg.lstsq(Ztr, ytr, rcond=None)
    return Zte @ beta


def _r2(y, pred, baseline):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss_tot = np.sum((y - baseline) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def cross_validate(X, y, country, rng):
    n = len(y)
    # random 10-fold: the interpolation bound
    idx = rng.permutation(n)
    p_kf = np.full(n, np.nan)
    for te in np.array_split(idx, KFOLDS):
        tr = np.setdiff1d(np.arange(n), te)
        p_kf[te] = _fit_predict(X[tr], y[tr], X[te])
    # leave-one-country-out: the transfer bound
    p_lo = np.full(n, np.nan)
    for c in pd.unique(country):
        te = country == c
        p_lo[te] = _fit_predict(X[~te], y[~te], X[te])

    gmean = y.mean()
    rows = []
    for c in pd.unique(country):
        m = country == c
        # A transferred relationship can fail two ways at once, and the single local R2
        # confounds them: it can put the whole country at the wrong LEVEL, or it can fail
        # to ORDER the catchments inside it. Splitting sum(r^2) = n*rbar^2 + sum((r-rbar)^2)
        # separates them; scoring only the second term against the same local baseline
        # gives a de-biased R2 that measures ordering alone. They have opposite remedies
        # (a few local gauges fix a level; nothing national fixes an ordering), so the
        # decomposition is reported rather than the local R2 alone.
        r = p_lo[m] - y[m]
        ss_loc = float(np.sum((y[m] - y[m].mean()) ** 2))
        rows.append({"country": c, "n": int(m.sum()),
                     "r2_global": _r2(y[m], p_lo[m], gmean),
                     "r2_local": _r2(y[m], p_lo[m], y[m].mean()),
                     "r2_local_debiased": (float(1 - np.sum((r - r.mean()) ** 2) / ss_loc)
                                           if ss_loc > 0 else np.nan),
                     # kept so the pooled de-biased score can be reassembled downstream
                     "ss_local": ss_loc,
                     "sse": float(np.sum(r ** 2)),
                     "sse_debiased": float(np.sum((r - r.mean()) ** 2)),
                     "rho_within": (float(stats.spearmanr(p_lo[m], y[m]).statistic)
                                    if m.sum() > 2 else np.nan),
                     "mean_bias": float((p_lo[m] - y[m]).mean())})
    per = pd.DataFrame(rows)
    big = per[per.n >= MIN_COUNTRY_N]
    # pooled local-baseline R2: every catchment scored against its own country's mean.
    # Its de-biased twin removes each country's mean residual first, so the pair isolates
    # the national level offset from the within-country ordering on the pooled scale too.
    local_base = pd.Series(y).groupby(pd.Series(country)).transform("mean").to_numpy()
    ss_loc_tot = float(per.ss_local.sum())
    pooled_debiased = (float(1 - per.sse_debiased.sum() / ss_loc_tot)
                       if ss_loc_tot > 0 else np.nan)
    return {
        "n": n, "n_countries": int(pd.Series(country).nunique()),
        "kfold_r2_global": _r2(y, p_kf, gmean),
        "loco_r2_global": _r2(y, p_lo, gmean),
        "loco_r2_local_pooled": _r2(y, p_lo, local_base),
        "loco_r2_local_debiased_pooled": pooled_debiased,
        "loco_r2_local_median_country": float(per.r2_local.median()),
        "loco_r2_local_wmean_country": float(np.average(per.r2_local, weights=per.n)),
        "loco_n_countries_negative_local": int((per.r2_local < 0).sum()),
        # decomposition, over countries large enough for a stable local score
        "loco_n_countries_big": int(len(big)),
        "loco_r2_local_median_big": float(big.r2_local.median()) if len(big) else np.nan,
        "loco_r2_local_debiased_median_big": (float(big.r2_local_debiased.median())
                                              if len(big) else np.nan),
        "loco_rho_within_median_big": (float(big.rho_within.median())
                                       if len(big) else np.nan),
        "loco_n_big_negative_local": int((big.r2_local < 0).sum()),
        "loco_n_big_negative_debiased": int((big.r2_local_debiased < 0).sum()),
        "loco_n_big_negative_rho": int((big.rho_within < 0).sum()),
    }, per


# --------------------------------------------------------------------------- driver

def analyse(d, response, label, nboot=NBOOT):
    sub = d.dropna(subset=[response]).copy()
    if len(sub) < 100 or sub.country.nunique() < 5:
        return None, None, None
    X_raw = sub[PREDICTORS].to_numpy(float)
    y = sub[response].to_numpy(float)
    country = sub["country"].to_numpy()
    Xz = (X_raw - X_raw.mean(0)) / np.where(X_raw.std(0) == 0, 1, X_raw.std(0))

    beta, se, t, r2, resid = ols_cluster(Xz, y, country)
    marg, partial, within = variance_split(Xz, y, country)
    icc = mom_icc(resid, country)
    cv, per = cross_validate(X_raw, y, country, np.random.default_rng(SEED))

    coefs = []
    for j, p in enumerate(PREDICTORS):
        pb, _ = wild_cluster_bootstrap(Xz, y, country, j, nboot=nboot)
        coefs.append({"response": response, "predictor": p, "beta_std": float(beta[j]),
                      "se_clustered": float(se[j]), "t": float(t[j]),
                      "p_wild_bootstrap": pb})
    # Six predictors are tested on every response, so the bootstrap p-values get the same
    # false-discovery-rate control already applied to the precipitation regression.
    for row, q in zip(coefs, bh_fdr([c["p_wild_bootstrap"] for c in coefs])):
        row["q_bh"] = float(q)
    summary = {"response": response, "label": label, "marginal_R2": marg,
               "country_partial_R2": partial, "predictor_partial_R2_within": within,
               "icc": icc, **cv}
    per.insert(0, "response", response)
    return pd.DataFrame(coefs), summary, per


def validate(summ: pd.DataFrame, coefs: pd.DataFrame):
    print("\n=== filter properties: fit, national structure, transfer ===", flush=True)
    hdr = (f"{'response':16s} {'n':>5s} {'margR2':>7s} {'within':>7s} {'ctryR2':>7s} "
           f"{'ICC':>6s} {'kfold':>7s} {'LOCOglob':>9s} {'LOCOlocal':>10s} {'medCtry':>8s} "
           f"{'neg':>4s}")
    print(hdr, flush=True)
    for _, r in summ.iterrows():
        print(f"{r['response']:16s} {int(r['n']):5d} {r['marginal_R2']:7.3f} "
              f"{r['predictor_partial_R2_within']:7.3f} {r['country_partial_R2']:7.3f} "
              f"{r['icc']:6.3f} {r['kfold_r2_global']:+7.3f} "
              f"{r['loco_r2_global']:+9.3f} {r['loco_r2_local_pooled']:+10.3f} "
              f"{r['loco_r2_local_median_country']:+8.3f} "
              f"{int(r['loco_n_countries_negative_local']):3d}/{int(r['n_countries']):d}",
              flush=True)
    print("\n  margR2   = six standardised predictors, cluster-robust OLS", flush=True)
    print("  within   = R2 the predictors ADD to country indicators alone "
          "(within-country signal)", flush=True)
    print("  LOCOglob = leave-one-country-out R2 vs the GLOBAL mean (flattering)", flush=True)
    print("  LOCOlocal= same predictions vs each held-out COUNTRY'S OWN mean", flush=True)
    print("  neg      = countries where the transferred fit loses to their own mean",
          flush=True)

    print("\n=== standardised coefficients (wild cluster bootstrap p) ===", flush=True)
    for resp, g in coefs.groupby("response", sort=False):
        sig = g[g.p_wild_bootstrap < 0.05]
        terms = ", ".join(f"{r.predictor} {r.beta_std:+.2f} (p={r.p_wild_bootstrap:.3f})"
                          for _, r in sig.iterrows()) or "none significant"
        print(f"  {resp:16s} {terms}", flush=True)


def selftest():
    rng = np.random.default_rng(0)
    # cluster-robust OLS recovers known slopes
    n, G = 1600, 16
    g = np.repeat([f"C{i}" for i in range(G)], n // G)
    X = rng.normal(size=(n, len(PREDICTORS)))
    truth = np.array([0.8, -0.5, 0.4, 0.0, 0.3, -0.2])
    y = 1.0 + X @ truth + np.repeat(rng.normal(0, 0.4, G), n // G) + rng.normal(0, 1.0, n)
    beta, se, t, r2, _ = ols_cluster(X, y, g)
    assert np.max(np.abs(beta - truth)) < 0.15, beta

    # the wild bootstrap must reject a real effect and keep a null one
    p_real, _ = wild_cluster_bootstrap(X, y, g, 0, nboot=399)
    p_null, _ = wild_cluster_bootstrap(X, y, g, 3, nboot=399)
    assert p_real < 0.05, p_real
    assert p_null > 0.10, p_null

    # and it must be more conservative than the normal reference under clustering
    from scipy import stats as st
    p_normal = 2 * (1 - st.norm.cdf(abs(t[3])))
    assert p_null >= p_normal - 0.05, (p_null, p_normal)

    # local vs global baseline: a fit that only gets country LEVELS right scores well
    # against the global mean and badly against each country's own mean
    lev = np.repeat(rng.normal(0, 3.0, G), n // G)
    y2 = lev + rng.normal(0, 1.0, n)
    pred = lev                                   # levels exactly, within-country nothing
    gmean = y2.mean()
    local = pd.Series(y2).groupby(pd.Series(g)).transform("mean").to_numpy()
    assert _r2(y2, pred, gmean) > 0.8, _r2(y2, pred, gmean)
    assert _r2(y2, pred, local) < 0.05, _r2(y2, pred, local)

    # variance_split: a response that is purely country-level has no within signal
    Xc = np.repeat(rng.normal(size=(G, len(PREDICTORS))), n // G, axis=0)
    y3 = Xc @ truth + rng.normal(0, 0.05, n)     # predictors constant within country
    m3, c3, w3 = variance_split(Xc, y3, g)
    assert m3 > 0.9 and w3 < 0.05, (m3, c3, w3)
    m4, c4, w4 = variance_split(X, y, g)          # varies within country -> real signal
    assert w4 > 0.2, (m4, c4, w4)

    # ICC of a pure country-level signal is near 1, of pure noise near 0
    assert mom_icc(lev + rng.normal(0, 0.01, n), g) > 0.9
    assert mom_icc(rng.normal(0, 1, n), g) < 0.1
    print("selftest OK  (cluster OLS, wild bootstrap conservative at G=16, "
          "local vs global baseline separate level from within-country skill)")


def main():
    ap = argparse.ArgumentParser()
    for f in ["strength-memory", "timing", "observed", "join", "signal", "attrs", "refined",
              "nesting", "outdir"]:
        ap.add_argument(f"--{f}")
    ap.add_argument("--independent-only", action="store_true")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all([a.strength_memory, a.timing, a.attrs, a.refined, a.outdir]):
        ap.error("--strength-memory --timing --attrs --refined --outdir required unless --selftest")

    strength_memory = pd.read_parquet(a.strength_memory)
    timing = pd.read_parquet(a.timing)
    observed = pd.read_parquet(a.observed) if a.observed else None
    nesting = pd.read_csv(a.nesting) if a.nesting else None
    gain_obs = None
    if a.join and a.signal:
        gain_obs = observed_gain(pd.read_parquet(a.join), pd.read_parquet(a.signal))
    d = prepare(strength_memory, timing, observed, gain_obs, pd.read_parquet(a.attrs),
                pd.read_csv(a.refined), nesting)
    if a.independent_only:
        if "independent" not in d.columns:
            ap.error("--independent-only needs --nesting")
        d = d[d["independent"].fillna(True)].reset_index(drop=True)
    print(f"analysis frame: {len(d)} catchments, {d.country.nunique()} countries"
          f"{'  [spatially independent subset]' if a.independent_only else ''}", flush=True)

    all_coefs, all_summ, all_per = [], [], []
    for resp, family, label, _tc in PROPERTIES:
        if resp not in d.columns:
            continue
        c, s, p = analyse(d, resp, label, nboot=a.nboot)
        if s is None:
            print(f"  skipped {resp} (too few usable catchments)", flush=True)
            continue
        s["family"] = family
        all_coefs.append(c)
        all_summ.append(s)
        all_per.append(p)
        print(f"  done {resp}  (n={s['n']})", flush=True)

    coefs = pd.concat(all_coefs, ignore_index=True)
    summ = pd.DataFrame(all_summ)
    per = pd.concat(all_per, ignore_index=True)
    tag = a.tag or ("independent" if a.independent_only else "full")
    for name, obj in [("coeffs", coefs), ("summary", summ), ("by_country", per)]:
        path = os.path.join(a.outdir, f"physiographic_{name}_{tag}.csv")
        obj.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)
    validate(summ, coefs)


if __name__ == "__main__":
    main()

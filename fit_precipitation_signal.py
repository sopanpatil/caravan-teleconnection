#!/usr/bin/env python
"""
fit_precipitation_signal.py

First half of the forcing/response decomposition: local meteorological
sensitivity. For each catchment, jointly regress the DJF precipitation anomaly on
the standardised CPC winter teleconnection indices

    precip_DJF_anom(c,wy) = b0 + b_NAO*NAO + b_EA*EA + b_EAWR*EAWR + b_SCA*SCA + e

The betas are local rainfall sensitivities (mm/day per unit index) and are what
response_strength_and_memory.py normalises by (the fitted index-explained signal P'_hat = X[1:] @ beta[1:]).
Per catchment we report each beta with OLS standard error / t / p, the joint model
R^2 / adj-R^2 / F p-value, and n winters. p-values are then FDR-controlled
(Benjamini-Hochberg) ACROSS catchments, separately per index, giving q-values and
significance flags -- the map-level multiple-comparison guard.

Output: one row per catchment with gauge_id, source, gauge_lat, gauge_lon, area,
n_winters, r2, adj_r2, f_pvalue, and beta/se/t/p/q/sig per index.

    python fit_precipitation_signal.py --join <seasonal_join_DJF.parquet> \
        --attrs <calibrated_parameters_ALL_refined.csv> --out <precipitation_signal_DJF.parquet>
    python fit_precipitation_signal.py --selftest
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
from scipy import stats

INDICES = ["NAO_DJF", "EA_DJF", "EAWR_DJF", "SCA_DJF"]
SHORT = {"NAO_DJF": "NAO", "EA_DJF": "EA", "EAWR_DJF": "EAWR", "SCA_DJF": "SCA"}
RESPONSE = "precip_DJF_anom"
MIN_WINTERS = 20


def fit_catchment(y: np.ndarray, Xp: np.ndarray) -> dict:
    """OLS with intercept. Xp = predictor matrix (n x k), no intercept column."""
    n, k = Xp.shape
    X = np.column_stack([np.ones(n), Xp])
    p = k + 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ssres = float(resid @ resid)
    sstot = float(((y - y.mean()) ** 2).sum())
    dof = n - p
    r2 = 1 - ssres / sstot if sstot > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
    sigma2 = ssres / dof if dof > 0 else np.nan
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        xtx_inv = np.full((p, p), np.nan)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), dof)
    ssreg = sstot - ssres
    F = (ssreg / k) / (ssres / dof) if (dof > 0 and ssres > 0) else np.nan
    f_p = float(stats.f.sf(F, k, dof)) if np.isfinite(F) else np.nan
    out = {"n_winters": n, "r2": r2, "adj_r2": adj_r2, "f_pvalue": f_p}
    for j, idx in enumerate(INDICES, start=1):  # skip intercept at 0
        s = SHORT[idx]
        out[f"beta_{s}"] = beta[j]
        out[f"se_{s}"] = se[j]
        out[f"t_{s}"] = t[j]
        out[f"p_{s}"] = pvals[j]
    return out


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaNs pass through as NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return q
    pp = p[idx]
    m = pp.size
    order = np.argsort(pp)
    ranked = pp[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    qv = np.empty(m)
    qv[order] = np.clip(ranked, 0, 1)
    q[idx] = qv
    return q


def run(join: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    cols = [RESPONSE] + INDICES
    rows = []
    for (gid, src), g in join.groupby(["gauge_id", "source"], sort=False):
        d = g[cols].dropna()
        if len(d) < MIN_WINTERS:
            continue
        rec = fit_catchment(d[RESPONSE].to_numpy(), d[INDICES].to_numpy())
        rec["gauge_id"] = gid
        rec["source"] = src
        rows.append(rec)
    res = pd.DataFrame(rows)

    for idx in INDICES:                      # FDR across catchments, per index
        s = SHORT[idx]
        res[f"q_{s}"] = bh_fdr(res[f"p_{s}"].to_numpy())
        res[f"sig_{s}"] = res[f"q_{s}"] < 0.05

    a = attrs.set_index("gauge_id")[["gauge_lat", "gauge_lon", "area"]]
    res = res.merge(a, left_on="gauge_id", right_index=True, how="left")
    front = ["gauge_id", "source", "gauge_lat", "gauge_lon", "area",
             "n_winters", "r2", "adj_r2", "f_pvalue"]
    return res[front + [c for c in res.columns if c not in front]]


def validate(res: pd.DataFrame):
    print("=== precipitation signal summary ===", flush=True)
    print(f"catchments fit: {len(res)}   median n_winters={int(res.n_winters.median())}"
          f"   median R2={res.r2.median():.3f}", flush=True)
    for s in ("NAO", "EA", "EAWR", "SCA"):
        frac = res[f"sig_{s}"].mean() * 100
        print(f"  beta_{s:4s}: median={res[f'beta_{s}'].median():+.3f} mm/d/unit   "
              f"FDR-sig (q<0.05): {frac:4.1f}%", flush=True)
    print("\n=== beta_NAO by latitude (recovers the dipole?) ===", flush=True)
    L = res.gauge_lat
    for lo, hi, name in [(35, 45, "Iberia 35-45N"), (45, 52, "mid 45-52N"),
                         (52, 58, "GB/DK 52-58N"), (58, 72, "boreal 58-72N")]:
        m = (L >= lo) & (L < hi)
        if m.any():
            print(f"  {name:16s} n={int(m.sum()):4d}  mean beta_NAO="
                  f"{res.loc[m,'beta_NAO'].mean():+.3f}  sig={res.loc[m,'sig_NAO'].mean()*100:4.1f}%",
                  flush=True)


def selftest():
    rng = np.random.default_rng(0)
    n = 60
    idx = rng.normal(size=(n, 4))
    true_b = np.array([2.0, -1.0, 0.0, 0.5])
    y = 3.0 + idx @ true_b + rng.normal(scale=0.5, size=n)
    rec = fit_catchment(y, idx)
    got = np.array([rec["beta_NAO"], rec["beta_EA"], rec["beta_EAWR"], rec["beta_SCA"]])
    assert np.allclose(got, true_b, atol=0.25), got
    assert rec["p_NAO"] < 1e-6 and rec["p_EAWR"] > 0.05, rec  # signal vs null
    # FDR monotone & bounded, and <= raw p elementwise for the largest, etc.
    p = np.array([0.001, 0.01, 0.04, 0.5, np.nan])
    q = bh_fdr(p)
    assert np.nanmax(q) <= 1 and np.all(np.diff(np.argsort(p[:4])) != 0)
    assert q[0] <= q[1] <= q[2] <= q[3] and np.isnan(q[4])
    print("selftest OK  (betas recovered, null insignificant, FDR monotone)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join")
    ap.add_argument("--attrs")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.join and a.attrs and a.out):
        ap.error("--join, --attrs and --out are required unless --selftest")
    join = pd.read_parquet(a.join)
    attrs = pd.read_csv(a.attrs)
    res = run(join, attrs)
    res.to_parquet(a.out)
    print(f"wrote {a.out}  ({len(res)} catchments)", flush=True)
    validate(res)


if __name__ == "__main__":
    main()

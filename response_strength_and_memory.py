#!/usr/bin/env python
"""
response_strength_and_memory.py

Stage-2 of the two-stage decomposition: how each catchment
*transforms* the teleconnection signal. Two per-catchment quantities per store
(SM soil moisture, SP snowpack, UZ upper zone, LZ lower zone) and for flow (Qsim):

  MEMORY -- each daily store is deseasonalised (minus its day-of-year climatology)
    and linearly detrended, then we report:
      tau_<S>  e-folding lag (days) of the autocorrelation -- the interpretable
               timescale. CENSORED (cens_<S>=True, tau=NaN) when the ACF has not
               fallen to 1/e within KMAX days: the store's memory exceeds what a
               34-70 yr record resolves. LZ (near-integrated at daily scale) is
               often censored -- that is the honest statement, not a made-up number.
      ac1_<S>  lag-1 autocorrelation of the same anomaly -- a bounded [0,1] memory
               proxy that is always defined and never blows up (unlike -1/ln(r1),
               which explodes for near-integrated stores). Use it for the synthesis
               where tau is censored. Higher ac1 = longer memory.
    tau(LZ) >> tau(SM) > tau(UZ) expected; tau(SP) meaningful only where snow is
    active (sp_active_frac).

  GAIN -- seasonal (DJF) store/flow anomaly per unit of the fitted
    rainfall signal P'_hat = sum_k beta_k * index_k (from fit_precipitation_signal). OLS
    slope of the store's DJF anomaly on P'_hat; normalising by P'_hat removes driver
    strength so gain is the per-unit response amplitude.

Feeds the physiographic synthesis (tau_flow ~ storage attributes + (1|country)).

    python response_strength_and_memory.py --states-dir <dir> --manifest <m.csv> \
        --join <seasonal_join_DJF.parquet> --signal <precipitation_signal_DJF.parquet> --out <strength_memory.parquet>
    python response_strength_and_memory.py --selftest
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd

STORES = ["SM", "SP", "UZ", "LZ", "Qsim"]
INDEX_BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
              "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}
KMAX = 1825          # e-folding search horizon (~5 yr); longer memory -> censored
MIN_DAYS = 3000      # need ~8+ yr of daily anomalies for a stable ACF


def deseasonalize_detrend(s: pd.Series) -> np.ndarray:
    """Remove day-of-year climatology, then a linear trend (fast, bincount-based)."""
    a = s.to_numpy(dtype=float)
    doy = s.index.dayofyear.to_numpy()
    fin = np.isfinite(a)
    sums = np.bincount(doy, weights=np.where(fin, a, 0.0), minlength=367)
    cnts = np.bincount(doy, weights=fin.astype(float), minlength=367)
    clim = sums / np.maximum(cnts, 1.0)
    a = a - clim[doy]
    if fin.sum() >= MIN_DAYS:
        t = np.arange(a.size, dtype=float)
        b1, b0 = np.polyfit(t[fin], a[fin], 1)
        a[fin] = a[fin] - (b0 + b1 * t[fin])
    return a


def lag1_autocorr(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < MIN_DAYS:
        return np.nan
    a = a - a.mean()
    d = float(np.dot(a, a))
    return float(np.dot(a[:-1], a[1:]) / d) if d > 0 else np.nan


def efolding_tau(a: np.ndarray, kmax: int = KMAX) -> float:
    """E-folding lag (days). np.inf = censored (memory > kmax); np.nan = degenerate."""
    a = a[np.isfinite(a)]
    n = a.size
    if n < MIN_DAYS:
        return np.nan
    a = a - a.mean()
    if np.dot(a, a) <= 0:
        return np.nan
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(a, m)
    acov = np.fft.irfft(f * np.conj(f))[: kmax + 1].real / n     # biased autocov
    acf = acov / acov[0]
    below = np.where(acf < 1.0 / np.e)[0]
    if below.size == 0:
        return np.inf                                            # censored
    k = int(below[0])
    if k == 0:
        return 0.0
    a0, a1 = acf[k - 1], acf[k]                                  # interp the crossing
    frac = (a0 - 1.0 / np.e) / (a0 - a1) if a0 != a1 else 0.0
    return (k - 1) + frac


def catchment_tau(df: pd.DataFrame) -> dict:
    out = {}
    for s in STORES:
        a = deseasonalize_detrend(df[s])
        tau = efolding_tau(a)
        out[f"tau_{s}"] = np.nan if not np.isfinite(tau) else tau
        out[f"cens_{s}"] = bool(np.isinf(tau))                    # memory > KMAX
        out[f"ac1_{s}"] = lag1_autocorr(a)
    sp = df["SP"].to_numpy(dtype=float)
    out["sp_active_frac"] = float(np.mean(sp > 0.1))
    return out


def catchment_gain(g: pd.DataFrame, betas: pd.Series) -> dict:
    """Seasonal gain of each store DJF anomaly on the fitted signal."""
    phat = sum(betas[INDEX_BETA[ix]] * g[ix] for ix in INDEX_BETA)
    out = {}
    for s in STORES:
        d = pd.concat([phat.rename("phat"), g[f"{s}_DJF_anom"].rename("y")], axis=1).dropna()
        if len(d) < 15 or d["phat"].var() == 0:
            out[f"gain_{s}"] = np.nan
        else:
            out[f"gain_{s}"] = float(np.polyfit(d["phat"], d["y"], 1)[0])
    out["n_winters_gain"] = int(np.isfinite(phat).sum())
    return out


def run(states_dir, manifest_path, join, signal):
    man = pd.read_csv(manifest_path).set_index("gauge_id")
    spinup = pd.to_datetime(man["spinup_end"])
    betas = signal.set_index("gauge_id")
    gain_groups = {gid: g for gid, g in join.groupby("gauge_id", sort=False)}

    rows = []
    files = sorted(glob.glob(os.path.join(states_dir, "*", "*.parquet")))
    for i, f in enumerate(files, 1):
        gid = os.path.basename(f)[:-len(".parquet")]
        src = os.path.basename(os.path.dirname(f))
        if gid not in betas.index:
            continue
        df = pd.read_parquet(f, columns=STORES)
        if gid in spinup.index:
            df = df[df.index >= spinup.loc[gid]]
        rec = {"gauge_id": gid, "source": src}
        rec.update(catchment_tau(df))
        if gid in gain_groups:
            rec.update(catchment_gain(gain_groups[gid], betas.loc[gid]))
        rows.append(rec)
        if i % 500 == 0:
            print(f"  {i}/{len(files)} catchments processed", flush=True)

    res = pd.DataFrame(rows)
    res = res.merge(betas[["gauge_lat", "gauge_lon", "area"]],
                    left_on="gauge_id", right_index=True, how="left")
    return res


def validate(res: pd.DataFrame):
    print("=== flow memory: tau (e-folding, days) + ac1 (lag-1), medians ===", flush=True)
    for s in STORES:
        tau = res[f"tau_{s}"].dropna()
        cens = res[f"cens_{s}"].mean() * 100
        print(f"  {s:4s}: tau median={tau.median():6.1f}d  IQR=[{tau.quantile(.25):.0f},"
              f"{tau.quantile(.75):.0f}]  censored(>{KMAX}d)={cens:4.1f}%   "
              f"ac1 median={res[f'ac1_{s}'].median():.3f}", flush=True)
    print(f"\n  ordering (ac1, robust): LZ {res.ac1_LZ.median():.3f} > SM "
          f"{res.ac1_SM.median():.3f} > UZ {res.ac1_UZ.median():.3f}", flush=True)
    snowy = res[res.sp_active_frac > 0.3]
    print(f"  tau_SP where snow active (>30% days, n={len(snowy)}): "
          f"median={snowy.tau_SP.median():.0f}d  ac1={snowy.ac1_SP.median():.3f}", flush=True)
    print("\n=== groundwater memory by latitude (LZ censored% = fraction with tau>5yr) ===", flush=True)
    L = res.gauge_lat
    for lo, hi, name in [(35, 45, "Iberia"), (45, 52, "mid"), (52, 58, "GB/DK"), (58, 72, "boreal")]:
        m = (L >= lo) & (L < hi)
        if m.any():
            print(f"  {name:8s} n={int(m.sum()):4d}  ac1_LZ={res.loc[m,'ac1_LZ'].median():.3f}"
                  f"  LZ censored={res.loc[m,'cens_LZ'].mean()*100:4.0f}%"
                  f"  ac1_SP={res.loc[m,'ac1_SP'].median():.3f}", flush=True)
    print("\n=== gain (DJF anomaly per unit P'_hat), median ===", flush=True)
    for s in STORES:
        print(f"  gain_{s:4s}: median={res[f'gain_{s}'].median():+.3f}", flush=True)


def selftest():
    rng = np.random.default_rng(0)
    # AR(1) with known tau: e-folding within 20% (under-estimates long tau);
    # lag-1 autocorr recovers r1 exactly.
    for tau_true in (10.0, 60.0, 300.0):
        n = 40000
        r1 = np.exp(-1.0 / tau_true)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = r1 * x[t - 1] + rng.normal(0, 1)
        assert abs(efolding_tau(x) - tau_true) / tau_true < 0.20, ("efold", tau_true, efolding_tau(x))
        assert abs(lag1_autocorr(x) - r1) < 0.01, ("ac1", r1, lag1_autocorr(x))
    # random walk -> censored (inf)
    rw = np.cumsum(rng.normal(size=20000))
    assert np.isinf(efolding_tau(rw)), efolding_tau(rw)
    # deseasonalise removes a pure annual cycle
    idx = pd.date_range("1990-01-01", periods=8000, freq="D")
    seas = pd.Series(5 * np.sin(2 * np.pi * idx.dayofyear / 365.25), index=idx)
    assert np.nanstd(deseasonalize_detrend(seas)) < 1e-6
    # gain recovers a known slope
    g = pd.DataFrame({k: rng.normal(size=40) for k in INDEX_BETA})
    g["SM_DJF_anom"] = 2.5 * g["NAO_DJF"] + rng.normal(0, .01, 40)
    for s in STORES:
        if s != "SM":
            g[f"{s}_DJF_anom"] = rng.normal(size=40)
    betas = pd.Series({"beta_NAO": 1.0, "beta_EA": 0.0, "beta_EAWR": 0.0, "beta_SCA": 0.0})
    assert abs(catchment_gain(g, betas)["gain_SM"] - 2.5) < 0.05
    print("selftest OK  (e-folding recovers AR1 10/60/300 within 20%, ac1 exact, "
          "random walk censored, deseasonalise clean, gain exact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--join")
    ap.add_argument("--signal")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all([a.states_dir, a.manifest, a.join, a.signal, a.out]):
        ap.error("--states-dir, --manifest, --join, --signal, --out required unless --selftest")
    join = pd.read_parquet(a.join)
    signal = pd.read_parquet(a.signal)
    res = run(a.states_dir, a.manifest, join, signal)
    res.to_parquet(a.out)
    print(f"wrote {a.out}  ({len(res)} catchments)", flush=True)
    validate(res)


if __name__ == "__main__":
    main()

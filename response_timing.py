#!/usr/bin/env python
"""
response_timing.py

Teleconnection-signal TIMING. A catchment filter has two independent properties: how LONG it
holds an anomaly (the memory tau) and WHEN it releases the signal (this script).
The memory estimate works on DJF flow only and is therefore blind to the snowmelt registration,
in which a winter teleconnection anomaly is stored as snow and re-emerges months
later at melt. Here we keep the FORCING strictly winter and track the RESPONSE across
the whole water year to locate that registration in time.

Method. For each catchment we build the fitted winter forcing
Phat'(y) = sum_k beta_k * index_k(y) (one value per winter-year), and we correlate it
with the catchment's monthly flow anomaly at every lag L = 0..11 months after the
December the winter begins (L=0 Dec(y-1), 1 Jan, 2 Feb, 3 Mar, ... 7 Jul, ...). The
lag profile r(L) is the signal's registration in time:

  * flashy      -> energy at L=0-2 (Dec-Feb); the winter signal is spent in winter.
  * groundwater -> sustained L=1-5; the aquifer releases it slowly.
  * snow        -> a second peak at L=6-8 (Jun-Aug): the winter signal re-emerges at melt.

From the noise-corrected energy profile w(L) = max(r(L)^2 - 1/(n-2), 0) we report a
matched pair (mirroring the interpretable-tau + bounded-ac1 design):

  reg_lag    energy-weighted centroid lag (months) -- INTERPRETABLE timing, sign-agnostic
             so the melt registration counts whatever its sign.
  late_frac  fraction of response energy at L>=4 (April onward) -- a BOUNDED [0,1]
             phase-shift index, robust for the physiography transfer test: ~0 flashy,
             high for snow.
  winter_r   r at the winter lags (L=0-2), the contemporaneous registration strength.
  sig        peak |r| over the window -- signal strength, to flag catchments where the
             teleconnection barely controls precipitation (e.g. the Alpine node) and the
             lag is therefore ill-defined.
  retention  share of the EARLY (Dec-Jun, L=0-6) response energy arriving in March or
             later, sum(w[3:7]) / sum(w[0:7]) -- how long the forced anomaly LASTS, as
             opposed to when it arrives. NaN where the early window carries no energy.

Why retention exists. Memory (the memory tau) is estimated from the flow series alone and
never refers to Phat', so it is an INTRINSIC catchment timescale; that it also governs
how long a teleconnection anomaly persists is a claim to be tested, not assumed. This
metric is that test. It is deliberately ONE-SIDED, cut off at L=6, because the obvious
two-sided alternative -- the energy-weighted spread of the whole profile about its
centroid -- is inflated by the separate melt peak that snow catchments carry at L~6-8,
which is the TIMING property, not the memory one. Measured that way the correlation
with tau comes out NEGATIVE (rho ~ -0.23 on simulated flow), an artefact of snow
catchments having both a late centroid and a wide profile. Measured one-sided on the
snow-free catchments, where the profile is unimodal and the two properties separate
cleanly, retention rises with memory at rho ~ +0.43 (and ~ +0.44 against the groundwater
timescale tau_LZ), against ~ +0.22 over the whole sample. See Text S2 of the
manuscript.

    python response_timing.py --states-dir <dir> --manifest <m.csv> \
        --signal <precipitation_signal_DJF.parquet> --indices <teleconnection_seasonal.csv> \
        --strength-memory <response_strength_memory_DJF.parquet> --out <response_timing_DJF.parquet>
    python response_timing.py --selftest
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import pandas as pd

BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
        "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}
LAGS = list(range(0, 12))     # L=0 Dec(y-1) ... L=11 Nov(y)
SIG_MIN = 0.2                 # peak |r| needed before a lag profile is usable
LATE = 4                      # L>=LATE (April onward) counts as delayed/phase-shifted
MIN_WINTERS = 20             # winters needed for a stable per-lag correlation
RET_FROM = 3                  # retention numerator starts at L=3 (March)
RET_TO = 6                    # retention window ends at L=6 (June), inclusive: one-sided,
                              # so the L~6-8 melt peak cannot inflate it (see docstring)


def monthly_anom(qsim: pd.Series) -> pd.Series:
    """Monthly-mean flow anomaly (minus month-of-year climatology). Qsim is a
    complete model series, so every calendar month is populated."""
    m = qsim.resample("MS").mean()
    clim = m.groupby(m.index.month).transform("mean")
    return m - clim


def lag_profile(ma: pd.Series, phat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """r(L) and per-lag n: correlate the winter forcing Phat'(y) with the monthly
    flow anomaly L months after Dec(y-1), across winter-years."""
    r = np.full(len(LAGS), np.nan)
    nL = np.zeros(len(LAGS), dtype=int)
    idx = {(t.year, t.month): v for t, v in ma.items()}
    for j, L in enumerate(LAGS):
        xs, ys = [], []
        for y, p in phat.items():
            if not np.isfinite(p):
                continue
            dt = pd.Timestamp(int(y) - 1, 12, 1) + pd.DateOffset(months=L)
            v = idx.get((dt.year, dt.month), np.nan)
            if np.isfinite(v):
                xs.append(v)
                ys.append(p)
        nL[j] = len(xs)
        if len(xs) >= MIN_WINTERS and np.std(xs) > 0 and np.std(ys) > 0:
            r[j] = np.corrcoef(xs, ys)[0, 1]
    return r, nL


def retention_ratio(w: np.ndarray, L: np.ndarray) -> float:
    """Share of the early (L=0..RET_TO) response energy arriving at L>=RET_FROM.

    One-sided by construction: the window stops at RET_TO so that a snow catchment's
    melt peak, which sits at the far end of the profile, cannot enter either the
    numerator or the denominator. NaN when the early window carries no energy at all,
    which is the honest answer for a catchment whose entire response is the melt peak.
    """
    early = w[L <= RET_TO].sum()
    if early <= 0:
        return np.nan
    return float(w[(L >= RET_FROM) & (L <= RET_TO)].sum() / early)


def metrics(r: np.ndarray, nL: np.ndarray) -> dict:
    nan = {"reg_lag": np.nan, "late_frac": np.nan, "winter_r": np.nan,
           "peak_lag": np.nan, "sig": np.nan, "retention": np.nan,
           "n_lags": int(np.isfinite(r).sum())}
    fin = np.isfinite(r)
    if fin.sum() < 6:
        return nan
    L = np.array(LAGS, dtype=float)
    # noise-corrected response energy: subtract the null variance of a correlation
    w = np.where(fin, r ** 2 - 1.0 / np.maximum(nL - 2, 1), 0.0)
    w = np.clip(w, 0, None)
    W = w.sum()
    if W <= 0:
        return nan
    reg_lag = float((L * w).sum() / W)
    late_frac = float(w[L >= LATE].sum() / W)
    win = fin & (L <= 2)
    winter_r = float(np.nanmean(r[win])) if win.any() else np.nan
    peak_lag = int(L[np.nanargmax(np.abs(np.where(fin, r, np.nan)))])
    sig = float(np.nanmax(np.abs(r[fin])))
    return {"reg_lag": reg_lag, "late_frac": late_frac, "winter_r": winter_r,
            "peak_lag": peak_lag, "sig": sig,
            "retention": retention_ratio(w, L), "n_lags": int(fin.sum())}


def run(states_dir, manifest_path, signal, indices, strength_memory):
    man = pd.read_csv(manifest_path).set_index("gauge_id")
    spinup = pd.to_datetime(man["spinup_end"])
    betas = signal.set_index("gauge_id")
    idx_cols = list(BETA)
    rows = []
    files = sorted(glob.glob(os.path.join(states_dir, "*", "*.parquet")))
    for i, f in enumerate(files, 1):
        gid = os.path.basename(f)[:-len(".parquet")]
        src = os.path.basename(os.path.dirname(f))
        if gid not in betas.index:
            continue
        b = betas.loc[gid]
        phat = sum(b[BETA[c]] * indices[c] for c in idx_cols).dropna()
        df = pd.read_parquet(f, columns=["Qsim"])
        if gid in spinup.index:
            df = df[df.index >= spinup.loc[gid]]
        r, nL = lag_profile(monthly_anom(df["Qsim"]), phat)
        rec = {"gauge_id": gid, "source": src}
        rec.update(metrics(r, nL))
        rows.append(rec)
        if i % 500 == 0:
            print(f"  {i}/{len(files)} catchments processed", flush=True)
    res = pd.DataFrame(rows)
    if strength_memory is not None:
        res = res.merge(strength_memory[["gauge_id", "tau_Qsim", "ac1_Qsim", "sp_active_frac",
                                "gauge_lat", "gauge_lon"]], on="gauge_id", how="left")
    return res


def validate(res: pd.DataFrame):
    print("=== registration lag (timing of the winter signal in flow) ===",
          flush=True)
    ok = res[res.sig > SIG_MIN]        # catchments where the teleconnection signal is present
    print(f"catchments: {len(res)}   with detectable signal (peak|r|>0.2): {len(ok)}",
          flush=True)
    for src, g in res.groupby("source"):
        gg = g[g.sig > SIG_MIN]
        print(f"  {src:9s} n={len(g):4d} (signal {len(gg):4d}): median reg_lag="
              f"{gg.reg_lag.median():.2f} mo  late_frac={gg.late_frac.median():.2f}  "
              f"winter_r={gg.winter_r.median():+.2f}", flush=True)
    if "sp_active_frac" in res.columns:
        from scipy import stats
        d = ok[["late_frac", "reg_lag", "sp_active_frac"]].dropna()
        print("\n=== phase-shift tracks snow? (signal catchments) ===", flush=True)
        print(f"  Spearman late_frac vs snow-active frac = "
              f"{stats.spearmanr(d.late_frac, d.sp_active_frac).statistic:+.3f}", flush=True)
        print(f"  Spearman reg_lag   vs snow-active frac = "
              f"{stats.spearmanr(d.reg_lag, d.sp_active_frac).statistic:+.3f}", flush=True)
        snowy = d[d.sp_active_frac > 0.3]
        flashy = d[d.sp_active_frac < 0.05]
        print(f"  snow-active (>30%%, n={len(snowy)}): median reg_lag={snowy.reg_lag.median():.2f} mo, "
              f"late_frac={snowy.late_frac.median():.2f}", flush=True)
        print(f"  snow-free  (<5%%,  n={len(flashy)}): median reg_lag={flashy.reg_lag.median():.2f} mo, "
              f"late_frac={flashy.late_frac.median():.2f}", flush=True)

    if "tau_Qsim" in res.columns:
        from scipy import stats
        print("\n=== does intrinsic memory govern how long the forced signal lasts? ===",
              flush=True)
        # Snow-free only: where the profile is unimodal, memory and timing separate.
        # See the module docstring for why the two-sided alternative fails here.
        # NOTE the subsetting criterion. This uses the MODEL's snow-active fraction,
        # which is what this script has to hand; the manuscript (Text S2) subsets on the
        # HydroATLAS snow-fraction attribute instead (<= 0.05). The two give the
        # same sign and monotonicity but different n and a different rho, so do not
        # cross-quote the number printed here as the manuscript's.
        sub = ok[["retention", "tau_Qsim", "sp_active_frac"]].dropna()
        free = sub[sub.sp_active_frac < 0.05]
        for lab, g in [("all           ", sub), ("model snow-free", free)]:
            if len(g) < 20:
                continue
            rho = stats.spearmanr(g.retention, np.log(g.tau_Qsim)).statistic
            print(f"  {lab} n={len(g):4d}: Spearman retention vs log tau = {rho:+.3f}",
                  flush=True)
        if len(free) >= 40:
            q = pd.qcut(free.tau_Qsim, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            med = free.groupby(q, observed=True).retention.median()
            print("  model snow-free retention by memory quartile: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in med.items()), flush=True)


def selftest():
    rng = np.random.default_rng(0)
    # Build a monthly series whose anomaly responds to a winter forcing at a known
    # set of lags, and check reg_lag/late_frac recover the injected timing.
    years = np.arange(1960, 2020)
    phat = pd.Series(rng.normal(size=len(years)), index=years)
    dates = pd.date_range("1959-12-01", "2019-12-01", freq="MS")

    def build(lag_weights):
        a = pd.Series(rng.normal(0, 0.2, len(dates)), index=dates)
        for y, p in phat.items():
            for L, wgt in lag_weights.items():
                dt = pd.Timestamp(int(y) - 1, 12, 1) + pd.DateOffset(months=L)
                if dt in a.index:
                    a.loc[dt] += wgt * p
        return a

    # flashy: response at Dec-Feb (L=0,1,2)
    r, nL = lag_profile(build({0: 1.0, 1: 1.0, 2: 0.8}), phat)
    mf = metrics(r, nL)
    assert mf["reg_lag"] < 1.5, mf
    assert mf["late_frac"] < 0.1, mf
    # snow: winter bump + strong melt at L=7 (Jul)
    r, nL = lag_profile(build({1: 0.6, 7: 1.2}), phat)
    ms = metrics(r, nL)
    assert ms["reg_lag"] > 4.0, ms
    assert ms["late_frac"] > 0.5, ms
    assert ms["reg_lag"] > mf["reg_lag"] and ms["late_frac"] > mf["late_frac"]

    # retention separates DURATION from TIMING, which is the whole point of the metric.
    # sustained: an aquifer releasing the same winter anomaly on into spring.
    r, nL = lag_profile(build({0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}), phat)
    mr = metrics(r, nL)
    assert mr["retention"] > 0.25, mr
    # flashy spends the anomaly inside winter -> little energy left after February
    assert mf["retention"] < 0.10, mf
    assert mr["retention"] > mf["retention"], (mr, mf)
    # the snow case is LATE but not LONG: its melt peak sits outside the L<=6 window,
    # so retention must NOT reward it the way late_frac does.
    assert ms["retention"] < mr["retention"], (ms, mr)
    assert ms["late_frac"] > mr["late_frac"], (ms, mr)
    # pure noise -> weak signal, energy correction drives late_frac/ reg_lag unstable
    # but sig should be small
    r, nL = lag_profile(pd.Series(rng.normal(size=len(dates)), index=dates), phat)
    assert metrics(r, nL)["sig"] < 0.4, metrics(r, nL)
    print(f"selftest OK  (flashy reg_lag={mf['reg_lag']:.2f}/late={mf['late_frac']:.2f} vs "
          f"snow reg_lag={ms['reg_lag']:.2f}/late={ms['late_frac']:.2f}; noise weak)")
    print(f"  retention: flashy={mf['retention']:.3f}  sustained={mr['retention']:.3f}  "
          f"snow={ms['retention']:.3f}  <- snow is LATE (late_frac={ms['late_frac']:.2f}) "
          f"but not LONG, which is what the one-sided window is for")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--signal")
    ap.add_argument("--indices")
    ap.add_argument("--strength-memory")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all([a.states_dir, a.manifest, a.signal, a.indices, a.out]):
        ap.error("--states-dir, --manifest, --signal, --indices, --out required unless --selftest")
    signal = pd.read_parquet(a.signal)
    indices = pd.read_csv(a.indices).set_index("winter_year")
    strength_memory = pd.read_parquet(a.strength_memory) if a.strength_memory else None
    res = run(a.states_dir, a.manifest, signal, indices, strength_memory)
    res.to_parquet(a.out)
    print(f"wrote {a.out}  ({len(res)} catchments)", flush=True)
    validate(res)


if __name__ == "__main__":
    main()

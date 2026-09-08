#!/usr/bin/env python
"""
interannual_persistence.py

Teleconnection-SIGNAL persistence. the intrinsic memory tau is
the store's GENERIC autocorrelation timescale -- it contains no teleconnection
information. This script instead measures how long the *teleconnection-attributable*
flow anomaly lingers ACROSS winters: a memoryless catchment responds only to this
winter's forcing, while an aquifer carries a wet NAO+ winter's anomaly into the
following winter(s) even when the next winter's NAO is neutral.

Metric. For each catchment we build the fitted forcing (the teleconnection-
attributable DJF precipitation anomaly)

    Phat'(y) = sum_k beta_k * index_k(y)          (beta_k from fit_precipitation_signal)

and fit, per response store/flow DJF anomaly Q'(y), a geometric (Koyck) distributed
lag with a single decay phi and a gain g:

    Q'(y) = g * sum_{k>=0} phi^k * Phat'(y-k) + e ,   0 <= phi < 1

The convolved regressor Z_phi(y) = sum_k phi^k Phat'(y-k) is formed by the recursion
Z(y) = Phat'(y) + phi*Z(y-1) (reset across winter-year gaps), so phi costs no extra
degrees of freedom -- for each trial phi we do a 2-parameter OLS of Q' on Z_phi and
pick the phi that maximises R^2. The persistence is then

    tc_tau  = -1/ln(phi)   winters   (linear-reservoir timescale of the SIGNAL)
    mean_lag = phi/(1-phi) winters

phi ~ 0  -> flashy, the signal is spent within its own winter; phi large -> the
teleconnection anomaly persists across winters. phi is capped at PHI_MAX (timescale
~6 winters); reaching the cap is flagged (tc_cens), the winter-resolution analogue of
the tau censoring -- a ~34-68 winter record cannot resolve arbitrarily long
interannual persistence.

Applied per store this shows the two reservoirs IN THE TELECONNECTION SIGNAL itself
(LZ carries phi>0 across winters; UZ/SM/SP do not), and tc_tau(Qsim) is cross-checked
against the intrinsic tau_Qsim / ac1_Qsim: if they agree, the intrinsic memory
IS the teleconnection-signal persistence, which justifies carrying tau downstream.

    python interannual_persistence.py --join <seasonal_join_DJF.parquet> \
        --signal <precipitation_signal_DJF.parquet> --strength-memory <response_strength_memory_DJF.parquet> --out <interannual_persistence_DJF.parquet>
    python interannual_persistence.py --selftest
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
from scipy import stats

# Response anomalies: flow first (the headline), then the stores for the
# two-reservoir demonstration in the teleconnection signal.
RESPONSES = ["Qsim", "LZ", "UZ", "SM", "SP"]
INDEX_BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
              "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}

PHI_GRID = np.linspace(0.0, 0.90, 46)   # 0.90 -> timescale -1/ln(.9) ~ 9.5 winters
PHI_MAX = PHI_GRID[-1]
BURN = 3            # drop the first BURN winters of each contiguous run (Koyck transient)
MIN_WINTERS = 25    # usable (post-burn) winters needed for a stable 2-param fit


def koyck_filter(phat: np.ndarray, years: np.ndarray, phi: float) -> np.ndarray:
    """Z(y) = Phat'(y) + phi*Z(y-1), reset whenever the winter-year step != 1.
    Returns the geometrically weighted history of the forcing at each winter."""
    z = np.empty_like(phat, dtype=float)
    for i in range(len(phat)):
        if i == 0 or years[i] - years[i - 1] != 1:
            z[i] = phat[i]
        else:
            z[i] = phat[i] + phi * z[i - 1]
    return z


def _burn_mask(years: np.ndarray, burn: int) -> np.ndarray:
    """True for rows kept: drop the first `burn` rows of each contiguous winter run
    (the Koyck recursion's pre-sample transient lives there)."""
    keep = np.ones(len(years), dtype=bool)
    run_start = 0
    for i in range(len(years)):
        if i == 0 or years[i] - years[i - 1] != 1:
            run_start = i
        if i - run_start < burn:
            keep[i] = False
    return keep


def fit_persistence(phat: np.ndarray, y: np.ndarray, years: np.ndarray) -> dict:
    """Grid-search phi for the geometric distributed lag Q' = g*Z_phi + e.
    phat, y, years are aligned, finite, sorted by winter_year."""
    nan = {"phi": np.nan, "tc_tau": np.nan, "mean_lag": np.nan,
           "gain0": np.nan, "r2": np.nan, "n": len(y), "tc_cens": False}
    if len(y) < MIN_WINTERS + BURN or np.var(phat) == 0 or np.var(y) == 0:
        return nan
    keep = _burn_mask(years, BURN)
    if keep.sum() < MIN_WINTERS:
        return nan
    yk = y[keep] - y[keep].mean()
    sst = float(yk @ yk)
    if sst <= 0:
        return nan
    best = (-np.inf, np.nan, np.nan)   # (r2, phi, g)
    for phi in PHI_GRID:
        z = koyck_filter(phat, years, phi)[keep]
        zc = z - z.mean()
        szz = float(zc @ zc)
        if szz <= 0:
            continue
        g = float(zc @ yk) / szz                       # OLS slope (centered)
        ssr = float(((yk - g * zc) ** 2).sum())
        r2 = 1.0 - ssr / sst
        if r2 > best[0]:
            best = (r2, phi, g)
    r2, phi, g = best
    tau = -1.0 / np.log(phi) if phi > 0 else 0.0
    return {"phi": phi, "tc_tau": tau, "mean_lag": phi / (1 - phi) if phi < 1 else np.inf,
            "gain0": g, "r2": r2, "n": int(keep.sum()), "tc_cens": bool(phi >= PHI_MAX)}


def run(join: pd.DataFrame, signal: pd.DataFrame, strength_memory: pd.DataFrame | None) -> pd.DataFrame:
    betas = signal.set_index("gauge_id")
    rows = []
    for (gid, src), g in join.groupby(["gauge_id", "source"], sort=False):
        if gid not in betas.index:
            continue
        b = betas.loc[gid]
        g = g.sort_values("winter_year")
        phat_full = sum(b[INDEX_BETA[ix]] * g[ix] for ix in INDEX_BETA)
        rec = {"gauge_id": gid, "source": src}
        for resp in RESPONSES:
            col = f"{resp}_DJF_anom"
            d = pd.concat([phat_full.rename("phat"), g[col].rename("y"),
                           g["winter_year"].rename("wy")], axis=1).dropna()
            f = fit_persistence(d["phat"].to_numpy(), d["y"].to_numpy(),
                                d["wy"].to_numpy())
            for k, v in f.items():
                rec[f"{k}_{resp}"] = v
        rows.append(rec)
    res = pd.DataFrame(rows)
    if strength_memory is not None:
        keep = ["gauge_id", "tau_Qsim", "ac1_Qsim", "tau_LZ", "ac1_LZ",
                "gauge_lat", "gauge_lon"]
        res = res.merge(strength_memory[keep], on="gauge_id", how="left")
    return res


def validate(res: pd.DataFrame):
    print("=== teleconnection-signal persistence ===", flush=True)
    print(f"catchments: {len(res)}", flush=True)
    for resp in RESPONSES:
        phi = res[f"phi_{resp}"].dropna()
        r2 = res[f"r2_{resp}"].dropna()
        cens = res[f"tc_cens_{resp}"].mean() * 100
        print(f"  {resp:4s}: n_fit={len(phi):4d}  median phi={phi.median():.3f}  "
              f"median tc_tau={np.nanmedian(res[f'tc_tau_{resp}']):.2f} winters  "
              f"median fit R2={r2.median():.3f}  censored(phi>={PHI_MAX:.2f})={cens:4.1f}%",
              flush=True)
    print("\n=== two reservoirs in the SIGNAL: fraction with phi>0.2 (persists >1 winter) ===",
          flush=True)
    for resp in RESPONSES:
        frac = (res[f"phi_{resp}"] > 0.2).mean() * 100
        print(f"  {resp:4s}: {frac:5.1f}%", flush=True)
    if "tau_Qsim" in res.columns:
        print("\n=== closing the loop: tc-persistence vs intrinsic memory ===",
              flush=True)
        for a, b, lab in [("tc_tau_Qsim", "tau_Qsim", "tc_tau(Qsim) vs intrinsic tau_Qsim"),
                          ("phi_Qsim", "ac1_Qsim", "phi(Qsim) vs intrinsic ac1_Qsim"),
                          ("tc_tau_LZ", "tau_LZ", "tc_tau(LZ)  vs intrinsic tau_LZ"),
                          ("phi_LZ", "ac1_LZ", "phi(LZ)   vs intrinsic ac1_LZ")]:
            d = res[[a, b]].dropna()
            if len(d) > 10:
                rho = stats.spearmanr(d[a], d[b]).statistic
                print(f"  Spearman {lab:42s} = {rho:+.3f}   (n={len(d)})", flush=True)


def selftest():
    rng = np.random.default_rng(0)
    # Synthetic catchment: Q'(y) = g * sum phi^k Phat'(y-k) + noise, known phi.
    for phi_true, g_true in [(0.0, 1.5), (0.5, 1.0), (0.75, 0.8)]:
        n = 200
        years = np.arange(1950, 1950 + n)
        phat = rng.normal(size=n)
        z = np.zeros(n)
        for t in range(1, n):
            z[t] = phat[t] + phi_true * z[t - 1]
        y = g_true * z + rng.normal(0, 0.05, n)
        f = fit_persistence(phat, y, years)
        assert abs(f["phi"] - phi_true) <= 0.06, (phi_true, f)
        assert abs(f["gain0"] - g_true) < 0.1, (g_true, f)
        assert f["r2"] > 0.95, f
    # Gap in the winter-year sequence must reset the recursion (no crash, still fits).
    n = 120
    years = np.concatenate([np.arange(1950, 1990), np.arange(1995, 2075)])[:n]
    phat = rng.normal(size=n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phat[t] + (0.6 * z[t - 1] if years[t] - years[t - 1] == 1 else 0.0)
    y = 1.0 * z + rng.normal(0, 0.05, n)
    f = fit_persistence(phat, y, years)
    assert abs(f["phi"] - 0.6) <= 0.08, f
    # Too-short record -> NaN, not a crash.
    assert np.isnan(fit_persistence(rng.normal(size=10), rng.normal(size=10),
                                    np.arange(10))["phi"])
    print("selftest OK  (phi recovered for 0.0/0.5/0.75, gain exact, gaps reset, "
          "short record -> NaN)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join")
    ap.add_argument("--signal")
    ap.add_argument("--strength-memory")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.join and a.signal and a.out):
        ap.error("--join, --signal and --out are required unless --selftest")
    join = pd.read_parquet(a.join)
    signal = pd.read_parquet(a.signal)
    strength_memory = pd.read_parquet(a.strength_memory) if a.strength_memory else None
    res = run(join, signal, strength_memory)
    res.to_parquet(a.out)
    print(f"wrote {a.out}  ({len(res)} catchments)", flush=True)
    validate(res)


if __name__ == "__main__":
    main()

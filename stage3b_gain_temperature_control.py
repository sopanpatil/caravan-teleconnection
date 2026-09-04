#!/usr/bin/env python
"""
stage3b_gain_temperature_control.py

The exclusion restriction behind the response strength (Text S1 of the manuscript).

The response strength ("gain") is the OLS slope of the DJF flow anomaly on the Stage-1
fitted forcing Phat'. Because Phat' is the projection of precipitation on the four
circulation indices, cov(P', Phat') = var(Phat'), so that slope is ALGEBRAICALLY the
two-stage least-squares estimate of dQ/dP instrumented by the indices. Reading it as
"the share of the winter precipitation anomaly that reaches the gauge" therefore carries
an exclusion restriction: the modes must reach winter flow ONLY through precipitation.

That restriction is not exactly satisfied. The same modes control winter temperature,
and hence precipitation phase, melt timing and evaporative demand. This script measures
the exposure and the damage:

  r_phat_that      within-catchment corr(Phat', That'), where That' is the analogous
                   fitted TEMPERATURE signal (DJF temperature anomaly regressed on the
                   same four indices). This is how entangled the two pathways are.
  gain_{obs,sim}   the published single-regressor gain, recomputed here so that the
                   controlled estimate is compared against an identically-constructed
                   baseline (validate() asserts it reproduces the archived columns).
  gain_{obs,sim}_tctrl
                   the gain refitted with That' entered alongside Phat', i.e. the partial
                   effect of the precipitation pathway holding temperature fixed.

Result: the pathways ARE entangled (median |r| ~ 0.45) but the estimate barely moves
(paired median change ~ -0.02), and the snow-free vs snow-active contrast that the
Results rest on survives. So the snow control on within-season transfer is a property of
the partition, not temperature loading onto a precipitation coefficient. The paper keeps
reporting the single-regressor gain, for comparability with the store-wise gains, which
are defined the same way.

Snow split. The PRIMARY split here is the model's snow-active fraction (sp_active_frac
>= 0.3), because this script exists to defend the 0.89-vs-0.17 contrast in Results
section 4.2.1 and must therefore use that section's own split, or it would be answering a
different question. A secondary split on the HydroATLAS attribute (frac_snow <= 0.05,
the criterion Text S2 uses for a different purpose -- isolating unimodal lag profiles) is
reported alongside, to show the conclusion does not depend on which is chosen.

    python stage3b_gain_temperature_control.py --derived <caravan_derived dir>
    python stage3b_gain_temperature_control.py --derived <dir> --out gain_tctrl.parquet
    python stage3b_gain_temperature_control.py --selftest
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import pandas as pd

MIN_WINTERS_GAIN = 20        # as stage3_full_synthesis.py
SP_ACTIVE = 0.3              # snowpack anomaly present in >=30%% of winters => snow-active.
                             # PRIMARY split: the same one Results sec. 4.2.1 uses for the
                             # 0.89-vs-0.17 contrast this script exists to defend.
SNOW_FREE = 0.05             # HydroATLAS frac_snow: secondary split, reported as a check
                             # that the conclusion is not an artefact of the primary one
BETA = {"NAO_DJF": "beta_NAO", "EA_DJF": "beta_EA",
        "EAWR_DJF": "beta_EAWR", "SCA_DJF": "beta_SCA"}
IDX = list(BETA)

# (label, value quoted in Text S1, tolerance)
EXPECTED = {
    "med_abs_r":        (0.45,   0.02),
    "iqr_lo_r":         (0.20,   0.02),
    "iqr_hi_r":         (0.73,   0.02),
    "shift_obs":        (-0.020, 0.005),
    "shift_sim":        (-0.015, 0.005),
    "snowfree_sim":     (0.89,   0.02),
    "snowfree_sim_t":    (0.86,   0.02),
    "snowactive_sim":   (0.17,   0.02),
    "snowactive_sim_t": (0.11,   0.02),
    "snowfree_obs":     (0.83,   0.02),
    "snowfree_obs_t":   (0.72,   0.02),
    "snowactive_obs":   (0.23,   0.02),
    "snowactive_obs_t": (0.18,   0.02),
}


def fitted_temperature(g: pd.DataFrame) -> np.ndarray:
    """That'(y): DJF temperature anomaly regressed on the same four circulation indices.

    The first stage for the pathway the exclusion restriction assumes away. Returns NaN
    where the record is too short to fit, so the controlled gain is simply not estimated
    there rather than being estimated from a degenerate fit.
    """
    d = g[IDX + ["temp_DJF_anom"]].dropna()
    if len(d) < MIN_WINTERS_GAIN:
        return np.full(len(g), np.nan)
    X = np.c_[np.ones(len(d)), d[IDX].to_numpy(float)]
    b = np.linalg.lstsq(X, d["temp_DJF_anom"].to_numpy(float), rcond=None)[0]
    return b[0] + g[IDX].to_numpy(float) @ b[1:]


def catchment_row(gid: str, g: pd.DataFrame, betas: pd.Series) -> dict:
    g = g.copy()
    g["phat"] = sum(betas[BETA[c]] * g[c] for c in BETA)
    g["that"] = fitted_temperature(g)
    out = {"gauge_id": gid}

    d = g[["phat", "that"]].dropna()
    out["r_phat_that"] = (float(np.corrcoef(d.phat, d.that)[0, 1])
                          if len(d) >= MIN_WINTERS_GAIN and d.that.var() > 0 else np.nan)

    for tag, col in [("obs", "flow_obs_DJF_anom"), ("sim", "Qsim_DJF_anom")]:
        # Single-regressor gain on [phat, y] ONLY, so it is the identical estimator to
        # the published gain. Dropping rows for a missing That' here would silently
        # refit the baseline on a smaller sample and make the shift uninterpretable.
        d = g[["phat", col]].dropna()
        out[f"n_winters_{tag}"] = len(d)
        if len(d) < MIN_WINTERS_GAIN or d["phat"].var() == 0:
            out[f"gain_{tag}"] = np.nan
        else:
            out[f"gain_{tag}"] = float(np.polyfit(d["phat"], d[col].to_numpy(float), 1)[0])

        # Controlled gain: the temperature pathway held fixed. Its own row set, its own n,
        # so validate() can confirm the two coincide and the paired shift is like-for-like.
        dt = g[["phat", "that", col]].dropna()
        out[f"n_winters_{tag}_tctrl"] = len(dt)
        if (len(dt) < MIN_WINTERS_GAIN or dt["phat"].var() == 0
                or dt["that"].var() == 0):
            out[f"gain_{tag}_tctrl"] = np.nan
        else:
            X = np.c_[np.ones(len(dt)), dt["phat"].to_numpy(float),
                      dt["that"].to_numpy(float)]
            out[f"gain_{tag}_tctrl"] = float(
                np.linalg.lstsq(X, dt[col].to_numpy(float), rcond=None)[0][1])
    return out


def run(join: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    b = stage1.set_index("gauge_id")
    rows = [catchment_row(gid, g, b.loc[gid])
            for gid, g in join.groupby("gauge_id", sort=False) if gid in b.index]
    return pd.DataFrame(rows)


def validate(res: pd.DataFrame, stage2: pd.DataFrame, attrs: pd.DataFrame) -> dict:
    d = (res.set_index("gauge_id")
            .join(stage2.set_index("gauge_id")[["gain_Qsim", "sp_active_frac"]])
            .join(attrs.set_index("gauge_id")[["frac_snow"]]))
    d = d.loc[d.index.intersection(stage2.gauge_id)]

    # The single-regressor column must reproduce the archived gain, or the controlled
    # estimate is being compared against a different baseline and the shift is meaningless.
    m = d[["gain_sim", "gain_Qsim"]].dropna()
    worst = float((m.gain_sim - m.gain_Qsim).abs().max())
    print(f"=== Text S1: the exclusion restriction on the response strength ===")
    print(f"catchments: {len(d)}")
    print(f"single-regressor gain reproduces archived gain_Qsim: max|diff| = {worst:.2e}"
          f"  (n={len(m)})")
    if worst > 1e-8:
        sys.exit("recomputed gain does not match the archived gain_Qsim; "
                 "the controlled comparison would not be like-for-like")

    r = d.r_phat_that.abs().dropna()
    out = {"med_abs_r": float(r.median()),
           "iqr_lo_r": float(r.quantile(.25)), "iqr_hi_r": float(r.quantile(.75))}
    print(f"\n|corr(Phat', That')|: median {out['med_abs_r']:.3f}  "
          f"IQR {out['iqr_lo_r']:.3f}-{out['iqr_hi_r']:.3f}   <- pathways entangled")

    mismatch = int(((d.n_winters_obs != d.n_winters_obs_tctrl)
                    | (d.n_winters_sim != d.n_winters_sim_tctrl)).sum())
    print(f"catchments where the controlled fit uses a different set of winters: "
          f"{mismatch}   (0 => the paired shift below is like-for-like)")

    print("\ngain, single-regressor -> temperature-controlled:")
    free, active = d[d.sp_active_frac < SP_ACTIVE], d[d.sp_active_frac >= SP_ACTIVE]
    for tag in ["obs", "sim"]:
        a, b = d[f"gain_{tag}"], d[f"gain_{tag}_tctrl"]
        ok = a.notna() & b.notna()
        out[f"shift_{tag}"] = float((b - a)[ok].median())
        print(f"  {tag}: median {a[ok].median():.3f} -> {b[ok].median():.3f}   "
              f"PAIRED median shift {out[f'shift_{tag}']:+.3f}  (n={int(ok.sum())})")
        for lab, sub in [("snow-free  ", free), ("snow-active", active)]:
            s = sub[[f"gain_{tag}", f"gain_{tag}_tctrl"]].dropna()
            key = ("snowfree" if "free" in lab else "snowactive") + f"_{tag}"
            out[key] = float(s[f"gain_{tag}"].median())
            out[key + "_t"] = float(s[f"gain_{tag}_tctrl"].median())
            print(f"     {lab} (sp_active_frac{'<' if 'free' in lab else '>='}{SP_ACTIVE}): "
                  f"{out[key]:.3f} -> {out[key + '_t']:.3f}   (n={len(s)})")
    print(f"\nsecondary split (HydroATLAS frac_snow <= {SNOW_FREE}), as a check:")
    for tag in ["obs", "sim"]:
        for lab, sub in [("snow-free  ", d[d.frac_snow <= SNOW_FREE]),
                         ("snow-active", d[d.frac_snow > SNOW_FREE])]:
            s = sub[[f"gain_{tag}", f"gain_{tag}_tctrl"]].dropna()
            print(f"  {tag} {lab}: {s[f'gain_{tag}'].median():.3f} -> "
                  f"{s[f'gain_{tag}_tctrl'].median():.3f}   (n={len(s)})")
    print("\n  -> entangled, but the estimate barely moves and the snow contrast holds")
    return out


def check(got: dict) -> None:
    bad = [f"  {k}: manuscript {w}, recomputed {got[k]:+.3f} (tolerance {t})"
           for k, (w, t) in EXPECTED.items() if abs(got[k] - w) > t]
    if bad:
        print("\nMISMATCH against the values quoted in Text S1:")
        print("\n".join(bad))
        sys.exit(1)
    print("OK: every value quoted in Text S1 reproduces from the derived products")


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 80
    g = pd.DataFrame({k: rng.normal(size=n) for k in IDX})
    betas = pd.Series({"beta_NAO": 0.5, "beta_EA": 0.2, "beta_EAWR": -0.3, "beta_SCA": 0.1})
    phat = sum(betas[BETA[c]] * g[c] for c in BETA)
    # temperature is driven by the SAME indices, so Phat' and That' are correlated,
    # and flow responds to BOTH pathways with known coefficients
    temp = 0.8 * g["NAO_DJF"] - 0.4 * g["EA_DJF"] + rng.normal(0, 0.05, n)
    g["temp_DJF_anom"] = temp
    that = np.linalg.lstsq(np.c_[np.ones(n), g[IDX].to_numpy()], temp.to_numpy(),
                           rcond=None)[0]
    that = that[0] + g[IDX].to_numpy() @ that[1:]
    truth_p, truth_t = 0.7, 0.9
    g["flow_obs_DJF_anom"] = truth_p * phat + truth_t * that + rng.normal(0, 0.02, n)
    g["Qsim_DJF_anom"] = g["flow_obs_DJF_anom"]

    row = catchment_row("test", g, betas)
    assert abs(row["r_phat_that"]) > 0.3, row          # the two pathways are entangled
    # controlling for temperature recovers the true precipitation coefficient;
    # the single-regressor gain is biased by the omitted temperature pathway
    assert abs(row["gain_obs_tctrl"] - truth_p) < 0.02, row
    assert abs(row["gain_obs"] - truth_p) > 0.05, row
    print(f"selftest OK  r(Phat,That)={row['r_phat_that']:+.3f}; "
          f"gain {row['gain_obs']:.3f} -> {row['gain_obs_tctrl']:.3f} "
          f"(truth {truth_p}), omitted-variable bias recovered")

    # no temperature record -> controlled gain not estimated, plain gain still is
    g2 = g.copy()
    g2["temp_DJF_anom"] = np.nan
    r2 = catchment_row("t2", g2, betas)
    assert np.isnan(r2["gain_obs_tctrl"]) and np.isfinite(r2["gain_obs"]), r2
    print("selftest OK  missing temperature degrades to the single-regressor gain only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="derived_data")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    d = a.derived.rstrip("/")
    res = run(pd.read_parquet(f"{d}/seasonal_join_DJF.parquet"),
              pd.read_parquet(f"{d}/stage1_DJF.parquet"))
    if a.out:
        res.to_parquet(a.out)
        print(f"wrote {a.out}  ({len(res)} catchments)\n")
    check(validate(res, pd.read_parquet(f"{d}/stage2_DJF.parquet"),
                   pd.read_parquet(f"{d}/attributes.parquet")))


if __name__ == "__main__":
    main()

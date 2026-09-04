"""
pet_penman_monteith.py

FAO-56 Penman-Monteith reference ET0 (Allen et al. 1998) computed from the
ERA5-Land variables that every Caravan-format dataset carries. We compute PET
ourselves, uniformly, for ALL countries so the PET method is held exactly fixed
across the sample: the forcing consistency the cross-national comparison rests on. Datasets that also ship
`potential_evaporation_sum_FAO_PENMAN_MONTEITH` (base Caravan v1.6, DK v7) are
used to VALIDATE this implementation; GRDC-Caravan v0.2 lacks that variable, so
for Fennoscandia this is the only source of a PM PET.

Caravan variable units (from the netCDF global `Units` attribute):
  surface_net_solar_radiation_*   W/m2      (daily mean)
  surface_net_thermal_radiation_* W/m2      (daily mean; net thermal is <=0)
  surface_pressure_*              kPa
  temperature_2m_*                deg C
  dewpoint_temperature_2m_*       deg C
  u/v_component_of_wind_10m_*     m/s (at 10 m)
Net radiation W/m2 -> MJ/m2/day: multiply by 86400/1e6 = 0.0864.
"""
from __future__ import annotations
import numpy as np
import xarray as xr

WM2_TO_MJ_M2_DAY = 0.0864          # W/m2 -> MJ/m2/day
WIND_10M_TO_2M = 4.87 / np.log(67.8 * 10.0 - 5.42)   # FAO-56 eq.47 (~0.748)


def _sat_vapour_pressure(t_c):
    """Saturation vapour pressure e0(T) [kPa], FAO-56 eq.11."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def penman_monteith_et0(t_mean, t_dew, rs_net_wm2, rl_net_wm2, u10, v10, p_kpa):
    """Penman-Monteith reference ET0 [mm/day] from daily ERA5-Land inputs.

    All inputs are 1-D arrays of equal length (daily). Soil heat flux G is taken
    as 0 at the daily step. Result is clipped at 0 (no negative demand).

    Saturation vapour pressure es is taken at T_mean (es = e0(T_mean)) rather
    than the FAO-56-recommended mean of e0(Tmax)/e0(Tmin). This matches the
    convention Caravan used for its own `potential_evaporation_sum_FAO_PENMAN_
    MONTEITH`: with this choice our ET0 reproduces Caravan's provided PM PET on
    DK to r=1.00000, bias ~-0.0001 mm/day. Using it uniformly (including for
    GRDC-Caravan, which ships no PM PET) holds the PET method exactly fixed
    across the whole sample -- the load-bearing consistency requirement.
    """
    t_mean = np.asarray(t_mean, float)
    # net radiation [MJ/m2/day]
    Rn = (np.asarray(rs_net_wm2, float) + np.asarray(rl_net_wm2, float)) * WM2_TO_MJ_M2_DAY
    G = 0.0

    # vapour pressures [kPa]: es at T_mean (Caravan convention), ea from dewpoint
    es = _sat_vapour_pressure(t_mean)
    ea = _sat_vapour_pressure(np.asarray(t_dew, float))
    vpd = np.clip(es - ea, 0.0, None)

    # slope of sat. vapour pressure curve [kPa/degC] (FAO-56 eq.13, at T_mean)
    delta = 4098.0 * _sat_vapour_pressure(t_mean) / (t_mean + 237.3) ** 2
    # psychrometric constant [kPa/degC] (FAO-56 eq.8)
    gamma = 0.000665 * np.asarray(p_kpa, float)

    # 2 m wind speed [m/s]
    u2 = np.hypot(np.asarray(u10, float), np.asarray(v10, float)) * WIND_10M_TO_2M

    num = 0.408 * delta * (Rn - G) + gamma * (900.0 / (t_mean + 273.0)) * u2 * vpd
    den = delta + gamma * (1.0 + 0.34 * u2)
    return np.clip(num / den, 0.0, None)


def et0_from_caravan(ds: xr.Dataset) -> np.ndarray:
    """Compute FAO-56 PM ET0 [mm/day] from an open Caravan basin Dataset."""
    g = lambda v: ds[v].values  # noqa: E731
    return penman_monteith_et0(
        t_mean=g("temperature_2m_mean"), t_dew=g("dewpoint_temperature_2m_mean"),
        rs_net_wm2=g("surface_net_solar_radiation_mean"),
        rl_net_wm2=g("surface_net_thermal_radiation_mean"),
        u10=g("u_component_of_wind_10m_mean"), v10=g("v_component_of_wind_10m_mean"),
        p_kpa=g("surface_pressure_mean"),
    )


if __name__ == "__main__":
    # Validate against a dataset that ships the provided FAO-PM PET (e.g. DK).
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python pet_penman_monteith.py <path to a Caravan basin .nc>\n"
                 "  needs a source that ships FAO-PM PET to validate against (e.g. DK)")
    path = sys.argv[1]
    ds = xr.open_dataset(path)
    ours = et0_from_caravan(ds)
    ref = ds["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].values
    m = np.isfinite(ours) & np.isfinite(ref)
    o, r = ours[m], ref[m]
    bias = np.mean(o - r)
    rmse = np.sqrt(np.mean((o - r) ** 2))
    corr = np.corrcoef(o, r)[0, 1]
    print(f"n={m.sum()}  ours_mean={o.mean():.4f}  ref_mean={r.mean():.4f}")
    print(f"bias(ours-ref)={bias:+.4f} mm/d   RMSE={rmse:.4f} mm/d   r={corr:.5f}")
    print(f"max|diff|={np.max(np.abs(o - r)):.4f} mm/d")

"""
caravan_io.py

Loader for Caravan-format per-basin netCDF timeseries, returning forcing in the
shape the hbv-model repo's HBVModel.run(precip, temp, evap) expects.

Caravan (and its extensions: DK, GRDC-Caravan, ...) share one standardised
schema. Variable names confirmed against the Caravan extension Denmark v7
netCDF files (2026-08). We deliberately use the dataset's own FAO-56
Penman-Monteith PET (added to base Caravan v1.6, DK v7, GRDC-Caravan) rather
than the ERA5-Land native `potential_evaporation_sum_ERA5_LAND`, which the
Caravan authors warn contains unrealistic values.

Forcing convention: precipitation, PET and streamflow are all mm/day;
temperature is degrees C. HBV consumes precip/temp/evap as daily arrays.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# --- Caravan variable names (one schema across base + all extensions) --------
PRECIP = "total_precipitation_sum"                         # mm/day
TEMP = "temperature_2m_mean"                               # deg C
PET = "potential_evaporation_sum_FAO_PENMAN_MONTEITH"      # mm/day (preferred)
PET_ERA5 = "potential_evaporation_sum_ERA5_LAND"           # mm/day (do NOT use)
FLOW = "streamflow"                                        # mm/day (observed)

# Optional independent storage observables (for validating HBV states later).
SWE = "snow_depth_water_equivalent_mean"                   # mm  -> validates SP
SOIL_LAYERS = [f"volumetric_soil_water_layer_{i}_mean" for i in range(1, 5)]  # -> validates SM


@dataclass
class BasinForcing:
    """Daily forcing + observed flow for one basin, aligned on a DatetimeIndex."""
    gauge_id: str
    dates: pd.DatetimeIndex
    precip: np.ndarray   # mm/day
    temp: np.ndarray     # deg C
    pet: np.ndarray      # mm/day (FAO-56 Penman-Monteith)
    flow: np.ndarray     # mm/day (observed; NaN where missing)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"precip": self.precip, "temp": self.temp, "pet": self.pet, "flow": self.flow},
            index=self.dates,
        )

    @property
    def n_flow_obs(self) -> int:
        return int(np.isfinite(self.flow).sum())


def basin_nc_path(root: str | Path, source: str, gauge_id: str) -> Path:
    """Path to a basin's netCDF, e.g. <root>/timeseries/netcdf/camelsdk/camelsdk_100006.nc.

    `gauge_id` may be given with or without the '<source>_' prefix.
    """
    root = Path(root)
    stem = gauge_id if gauge_id.startswith(f"{source}_") else f"{source}_{gauge_id}"
    return root / "timeseries" / "netcdf" / source / f"{stem}.nc"


def load_basin(nc_path: str | Path, *, pet_source: str = "computed") -> BasinForcing:
    """Load one Caravan basin netCDF into a BasinForcing.

    pet_source:
      "computed" (default) - FAO-56 PM PET computed from the ERA5-Land inputs by
          pet_penman_monteith. Works for every source (GRDC-Caravan ships no PM
          PET) and holds the PET method identical across the whole sample. It
          reproduces the Caravan-provided PM PET exactly where that exists (DK,
          base-Caravan GB): r=1.0, so "computed" == "provided" numerically there.
      "provided" - read `potential_evaporation_sum_FAO_PENMAN_MONTEITH` (QA only;
          absent in GRDC-Caravan).
      "era5"     - native ERA5-Land `pev` (discouraged; unrealistic values).

    Raises KeyError if a required variable is absent so callers fail loudly.
    """
    from pet_penman_monteith import et0_from_caravan  # local import: optional dep

    nc_path = Path(nc_path)
    with xr.open_dataset(nc_path) as ds:
        for v in (PRECIP, TEMP, FLOW):
            if v not in ds.variables:
                raise KeyError(f"{nc_path.name}: missing expected variable '{v}'")
        dates = pd.DatetimeIndex(pd.to_datetime(ds["date"].values))
        precip = np.asarray(ds[PRECIP].values, dtype=np.float64)
        temp = np.asarray(ds[TEMP].values, dtype=np.float64)
        flow = np.asarray(ds[FLOW].values, dtype=np.float64)

        if pet_source == "computed":
            pet = np.asarray(et0_from_caravan(ds), dtype=np.float64)
        elif pet_source in ("provided", "era5"):
            pet_var = PET if pet_source == "provided" else PET_ERA5
            if pet_var not in ds.variables:
                raise KeyError(f"{nc_path.name}: missing '{pet_var}' (pet_source={pet_source!r})")
            pet = np.asarray(ds[pet_var].values, dtype=np.float64)
        else:
            raise ValueError(f"unknown pet_source {pet_source!r}")

    gauge_id = nc_path.stem
    # PET can carry small negatives at high latitude in winter; clip at 0.
    pet = np.where(np.isfinite(pet), np.clip(pet, 0.0, None), pet)
    return BasinForcing(gauge_id=gauge_id, dates=dates, precip=precip, temp=temp, pet=pet, flow=flow)


if __name__ == "__main__":
    import sys

    # Smoke test against a Caravan netCDF given on the command line.
    if len(sys.argv) < 2:
        sys.exit("usage: python caravan_io.py <path to a Caravan basin .nc>\n"
                 "  e.g. <caravan>/timeseries/netcdf/camelsdk/camelsdk_100006.nc")
    path = sys.argv[1]
    bf = load_basin(path)
    print(f"gauge_id       : {bf.gauge_id}")
    print(f"dates          : {bf.dates.min().date()} -> {bf.dates.max().date()} ({len(bf.dates)} days)")
    print(f"flow obs (days): {bf.n_flow_obs}")
    for name, arr in [("precip", bf.precip), ("temp", bf.temp), ("pet", bf.pet), ("flow", bf.flow)]:
        finite = arr[np.isfinite(arr)]
        print(f"{name:7s}: mean={finite.mean():7.3f}  min={finite.min():7.3f}  max={finite.max():7.3f}")

#!/usr/bin/env python
"""
build_attributes.py

Assemble attributes.parquet: the climate signatures and physiography that the
physiographic synthesis regresses the three response properties on.

Twelve columns, pulled straight from the attribute CSVs the Caravan collections
ship beside their timeseries. Nothing is recomputed here; the point of the
script is that the provenance of every predictor is a named source column
rather than an undocumented working file.

  From attributes_caravan_<source>.csv (Caravan's own climate signatures):
    aridity         PET/P, long-term mean
    moisture_index  Caravan's moisture index
    seasonality     precipitation seasonality
    frac_snow       fraction of precipitation falling as snow
    p_mean          mean daily precipitation (mm/day)

  From attributes_hydroatlas_<source>.csv (HydroATLAS, catchment averages):
    gwt_depth       gwt_cm_sav   groundwater table depth (cm)
    karst_pc        kar_pc_sse   karst extent (%)
    slope_deg       slp_dg_sav   terrain slope (HydroATLAS degrees x 10)
    elev_m          ele_mt_sav   elevation (m)
    clay_pc         cly_pc_sav   clay fraction (%)
    sand_pc         snd_pc_sav   sand fraction (%)
    snowcov_pc      snw_pc_syr   mean annual snow cover extent (%)

One choice worth stating. The base Caravan and CAMELS-DK collections publish
aridity, moisture_index and seasonality twice, once computed from ERA5-Land and
once from FAO Penman-Monteith PET. We take the ERA5-Land variant, so that these
signatures rest on the same forcing as the model runs. The GRDC-Caravan
extension publishes the columns unsuffixed and they are read as they are.

The output covers the whole downloaded domain (7,195 catchments), not the
2,135-catchment analysis set: screening happens downstream, in
screen_analysis_sample.py, so that the screen stays reversible.

    python build_attributes.py --raw <caravan_raw/> --out attributes.parquet
    python build_attributes.py --selftest
"""
from __future__ import annotations
import argparse
import glob
import os

import pandas as pd

# HydroATLAS source column -> the name we use.
HYDROATLAS = {
    "gwt_cm_sav": "gwt_depth",
    "kar_pc_sse": "karst_pc",
    "slp_dg_sav": "slope_deg",
    "ele_mt_sav": "elev_m",
    "cly_pc_sav": "clay_pc",
    "snd_pc_sav": "sand_pc",
    "snw_pc_syr": "snowcov_pc",
}
# Caravan signatures published in an ERA5-Land and an FAO Penman-Monteith
# flavour; we want ERA5-Land, and fall back to the unsuffixed name that the
# GRDC-Caravan extension uses.
DUAL = ["aridity", "moisture_index", "seasonality"]
PLAIN = ["frac_snow", "p_mean"]

COLUMNS = ["gauge_id"] + DUAL + PLAIN + list(HYDROATLAS.values())

# The order the collections were concatenated when the archive was built. Fixed
# here so a rebuild reproduces derived_data/attributes.parquet exactly, rather
# than inheriting whatever order the filesystem happens to glob in.
SOURCE_ORDER = ["camelsdk", "camelsgb", "lamah", "grdc"]


def _read_ids(path: str) -> pd.DataFrame:
    """Read an attribute CSV, naming its first column gauge_id whatever it is."""
    d = pd.read_csv(path)
    return d.rename(columns={d.columns[0]: "gauge_id"})


def caravan_columns(d: pd.DataFrame) -> dict[str, str]:
    """Pick the ERA5-Land signature where it exists, else the plain name."""
    out = {}
    for k in DUAL:
        era5 = f"{k}_ERA5_LAND"
        out[era5 if era5 in d.columns else k] = k
    for k in PLAIN:
        out[k] = k
    return out


def build_source(attr_dir: str) -> pd.DataFrame:
    """Join one collection's Caravan and HydroATLAS attribute tables."""
    car = glob.glob(os.path.join(attr_dir, "attributes_caravan_*.csv"))
    hyd = glob.glob(os.path.join(attr_dir, "attributes_hydroatlas_*.csv"))
    if not car or not hyd:
        raise FileNotFoundError(f"missing attribute CSVs under {attr_dir}")
    c, h = _read_ids(car[0]), _read_ids(hyd[0])
    cmap = caravan_columns(c)
    c = c[["gauge_id"] + list(cmap)].rename(columns=cmap)
    h = h[["gauge_id"] + list(HYDROATLAS)].rename(columns=HYDROATLAS)
    return c.merge(h, on="gauge_id", how="inner")


def build(raw_root: str) -> pd.DataFrame:
    """Every collection found beneath raw_root, concatenated in SOURCE_ORDER.

    The order is fixed rather than filesystem-dependent so that a rebuild
    reproduces the archived attributes.parquet row for row. Any collection not
    named in SOURCE_ORDER is appended afterwards, sorted by name.
    """
    dirs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in glob.glob(os.path.join(raw_root, "**", "attributes_caravan_*.csv"),
                               recursive=True)}
    if not dirs:
        raise FileNotFoundError(
            f"no attributes_caravan_*.csv under {raw_root}; is that the raw Caravan root?")
    named = [dirs[s] for s in SOURCE_ORDER if s in dirs]
    rest = [dirs[s] for s in sorted(dirs) if s not in SOURCE_ORDER]
    out = pd.concat([build_source(d) for d in named + rest], ignore_index=True)
    return out[COLUMNS].reset_index(drop=True)


def validate(d: pd.DataFrame) -> None:
    assert list(d.columns) == COLUMNS, d.columns
    assert d.gauge_id.is_unique, "duplicate gauge_id"
    assert not d[DUAL + PLAIN].isna().all().any(), "a climate signature is entirely missing"
    print(f"  {len(d)} catchments, {len(d.columns)} columns, "
          f"{d.gauge_id.str.split('_').str[0].nunique()} sources", flush=True)


def selftest() -> None:
    """Synthetic collections: one publishing ERA5-Land columns, one not."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for src, dual in [("alpha", True), ("beta", False)]:
            d = os.path.join(tmp, src)
            os.makedirs(d)
            ids = [f"{src}_1", f"{src}_2"]
            car = {"gauge_id": ids, "frac_snow": [0.1, 0.2], "p_mean": [2.0, 3.0]}
            for k in DUAL:
                # the FAO variant carries a wrong value, so picking it would show
                car[f"{k}_ERA5_LAND" if dual else k] = [1.0, 2.0]
                if dual:
                    car[f"{k}_FAO_PM"] = [-99.0, -99.0]
            pd.DataFrame(car).to_csv(os.path.join(d, f"attributes_caravan_{src}.csv"), index=False)
            hyd = {"gauge_id": ids}
            for i, k in enumerate(HYDROATLAS):
                hyd[k] = [float(i), float(i + 1)]
            pd.DataFrame(hyd).to_csv(os.path.join(d, f"attributes_hydroatlas_{src}.csv"), index=False)

        got = build(tmp)
        assert len(got) == 4, got
        assert list(got.columns) == COLUMNS
        # the ERA5-Land value won wherever both were published
        assert (got.aridity == 1.0).sum() == 2 and (got.aridity == -99.0).sum() == 0
        assert got.gwt_depth.tolist() == [0.0, 1.0, 0.0, 1.0]
        validate(got)
    print("selftest OK  (ERA5-Land preferred, unsuffixed fallback read, sources concatenated)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.environ.get("CARAVAN_DEST", "caravan_raw"),
                    help="raw Caravan root (default $CARAVAN_DEST, else ./caravan_raw)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.out:
        ap.error("--out required unless --selftest")
    d = build(a.raw)
    validate(d)
    d.to_parquet(a.out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

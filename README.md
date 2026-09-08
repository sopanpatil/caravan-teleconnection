# caravan-teleconnection

Analysis pipeline for *"Catchment storage controls on the response of European
winter streamflow to North Atlantic teleconnections"*.

The study separates the climate forcing from the catchment response across
2,135 European catchments in 16 countries, taken from the
[Caravan](https://doi.org/10.1038/s41597-023-01975-w) community large-sample
dataset and forced consistently from ERA5-Land. It regresses winter
precipitation on four North Atlantic circulation modes to isolate the forcing,
drives a per-catchment calibrated HBV model to measure three response
properties (within-season transfer strength, storage memory, snowmelt timing),
recomputes all three directly on observed discharge, and then asks how far the
resulting physiographic relationships travel by withholding whole countries.

## Requirements

`pip install -r requirements.txt`, plus the HBV model itself, which lives in a
separate repository and is not on PyPI:

```
git clone https://github.com/sopanpatil/hbv-model
export HBV_MODEL_REPO=/path/to/hbv-model        # or put it on PYTHONPATH
```

`cartopy` is optional: it adds coastlines and borders to the map panels, and
every map falls back to a plain lon/lat frame without it.

## Paths

No path is baked in. The stage scripts take every input and output as an
explicit argument with no default, so a run states exactly which files it
consumed. Two environment variables cover the rest:

```
export HBV_MODEL_REPO=/path/to/hbv-model     # or put that checkout alongside
                                             # this one, which is the fallback
export CARAVAN_DEST=/path/for/raw/caravan    # default ./caravan_raw
```

The two Supporting Information verification scripts default to `--derived
derived_data`, so they run correctly in a fresh clone with no arguments.

## Data

Three tiers, of which the smallest and most important is carried here:

| Tier | Where it comes from |
|---|---|
| Raw Caravan (~57 GB) | Download with `download_caravan.sh`. Distributed by the Caravan authors and its extension authors; not redistributed here. |
| Daily HBV states (~7 GB) | Regenerate with `generate_states.py`. Working intermediate, not carried. |
| **Derived per-catchment products (~29 MB)** | **Tracked, in [`derived_data/`](derived_data/).** Every number in the Results and in SI Tables S1-S3 comes from these, so a clone reproduces the paper without the 57 GB download or the calibration run. |

One small input is also tracked, because it is cheap and pins the analysis:
`teleconnection_seasonal.csv`, the CPC indices as fetched and seasonally
averaged. `dk_flow_availability.csv` sits alongside it as provenance rather
than as an input, and no script reads it: it records the daily-record coverage
of all 308 Danish gauges that were calibrated, which is the audit behind the
Danish sample, 263 of which pass the screen into the analysis set.

Note the licence split: the code is MIT, the data in `derived_data/` are
CC BY 4.0. See `LICENSE` and `derived_data/README.md`.

## Pipeline

Run in order. Each stage script takes explicit input and output paths
(`--join`, `--stage1`, `--stage2`, `--out`, ...) and has no defaults, so pass
what you want; `--help` lists them. The two verification scripts take a single
`--derived <dir>`, defaulting to `derived_data`.

**Calibration**

| Script | Role |
|---|---|
| `download_caravan.sh` | Fetch Caravan base (which carries the CAMELS-GB and LamaH-CE sources) plus the CAMELS-DK and GRDC-Caravan extensions |
| `caravan_io.py` | Loader for the Caravan per-basin netCDF schema |
| `pet_penman_monteith.py` | FAO-56 Penman-Monteith PET, computed uniformly for every country |
| `calibrate_single_caravan.py`, `run_calibration_batch.py` | Per-catchment SCE-UA calibration of HBV against observed discharge |
| `aggregate_results.py` | Collect the per-basin calibration JSONs into `calibrated_parameters_ALL.csv` |
| `refine_calibration.py` | Adds attributes, the glacier screen and the LamaH/GRDC de-duplication as reversible flags -> `calibrated_parameters_ALL_refined.csv` |
| `generate_states.py` | Drive each calibrated model with its full daily forcing; save every internal store |

**Forcing and response properties**

| Script | Writes | Backs |
|---|---|---|
| `fetch_teleconnection_indices.py` | `teleconnection_seasonal.csv` | The four CPC circulation indices: NAO, EA, EAWR, SCA |
| `build_seasonal_join.py` | `seasonal_join_DJF.parquet` | Daily states collapsed to seasonal means and anomalies |
| `stage1_sensitivity.py` | `stage1_DJF.parquet` | Section 4.1, Figure 1 |
| `stage2_filtering.py` | `stage2_DJF.parquet` | Store memory and gains, Section 4.2, Figure 2 |
| `stage2c_registration_lag.py` | `stage2c_DJF.parquet` | Timing, Section 4.2.3, Figure 3 |
| `stage2_obs_validation.py` | `stage2_obs_DJF.parquet` | All three properties on observed flow, Section 4.3, Figure 4 |
| `stage2b_tc_persistence.py` | `stage2b_DJF.parquet` | The interannual-persistence property, tested and not carried forward |

**Synthesis**

| Script | Writes | Backs |
|---|---|---|
| `nesting_screen.py` | `nesting_flags.csv`, `nesting_stem_pairs.csv` | Section 2.3 and SI Text S3 |
| `stage3_full_synthesis.py` | `stage3_full_{coeffs,summary,by_country}_{full,independent}.csv` | Sections 4.4 to 4.6, Figure 5, SI Tables S1-S3 |
| `stage3b_gain_temperature_control.py` | `gain_temperature_control.parquet` | SI Text S1 |
| `verify_text_s2.py` | nothing | Regression check on the values quoted in SI Text S2 |

**Figures**

Each figure script lives in `figures/` and is named for the figure it writes,
alongside that figure's `.png` and `.pdf`. Run them from the repository root;
with no `--out` each writes its published path.

| Script | Figure |
|---|---|
| `figures/fig1_nao_precipitation_sensitivity.py` | Figure 1 |
| `figures/fig2_store_memory.py` | Figure 2 |
| `figures/fig3_snowmelt_timing.py` | Figure 3 |
| `figures/fig4_fig5_validation_and_transfer.py` | Figures 4 and 5 (`fig4_observed_validation`, `fig5_physiography_and_transfer`) |

## Reproducing from `derived_data/`

The tracked products are enough to rebuild the whole synthesis, with no raw
download and no calibration run.

One trap worth naming: `--attrs` means a different file in two scripts. In
`stage1_sensitivity.py` it is the calibration table, which carries the gauge
coordinates; in `stage3_full_synthesis.py` it is `attributes.parquet`, the
climate and physiography predictors.

Stage 1, which reproduces `stage1_DJF.parquet` exactly:

```
python stage1_sensitivity.py \
    --join  derived_data/seasonal_join_DJF.parquet \
    --attrs derived_data/calibrated_parameters_ALL_refined.csv \
    --out   stage1_DJF.parquet
```

The synthesis, which regenerates SI Tables S1 to S3:

```
python stage3_full_synthesis.py \
    --join      derived_data/seasonal_join_DJF.parquet \
    --stage1    derived_data/stage1_DJF.parquet \
    --stage2    derived_data/stage2_DJF.parquet \
    --stage2c   derived_data/stage2c_DJF.parquet \
    --stage2obs derived_data/stage2_obs_DJF.parquet \
    --attrs     derived_data/attributes.parquet \
    --refined   derived_data/calibrated_parameters_ALL_refined.csv \
    --nesting   derived_data/nesting_flags.csv \
    --outdir    out --nboot 1999 --tag full
```

Add `--independent-only --tag independent` for the spatially independent
subset. `--nboot 1999` is the published setting and takes a while; a smaller
value gives the same coefficients with coarser bootstrap p-values.

The three response properties themselves cannot be recomputed from
`derived_data/` alone: `stage2_filtering.py`, `stage2c_registration_lag.py`
and `stage2_obs_validation.py` read the daily HBV state series, about 7 GB,
which is not carried here because it regenerates from `generate_states.py`.
Their outputs are archived instead. Figure 3 is in the same position.

## Verifying the reported numbers

Most stage scripts carry a `--selftest` that runs on synthetic data with a
known answer and needs no archive:

```
for f in stage1_sensitivity stage2_filtering stage2c_registration_lag \
         stage2_obs_validation stage3_full_synthesis nesting_screen \
         stage2b_tc_persistence stage3b_gain_temperature_control \
         build_seasonal_join generate_states refine_calibration; do
    python $f.py --selftest || echo "FAILED: $f"
done
```

`verify_manuscript.py` re-derives every quantitative claim in the paper that
the archived products can support, and checks every parameter the Methods
states against the constant in this code that implements it. It exits non-zero
on any mismatch:

```
python verify_manuscript.py                  # 109 claims + 17 parameters
python verify_manuscript.py --list-unchecked # what it does not cover, and why
```

Two further scripts cover the Supporting Information text sections, and fail
loudly the same way:

```
python stage3b_gain_temperature_control.py   # SI Text S1
python verify_text_s2.py                     # SI Text S2
```

Supporting Information Tables S1 to S3 are rendered directly from
`derived_data/stage3_full_*.csv`, so they are checked by construction.

## Licence and citation

Code: MIT (see `LICENSE`). Data in `derived_data/`: CC BY 4.0. If you use
either, cite the software and the paper, as `CITATION.cff` sets out.

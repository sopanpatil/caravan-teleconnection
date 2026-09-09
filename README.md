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
recomputes all three directly on observed discharge, and then tests how far the
resulting physiographic relationships transfer by withholding whole countries.

## Requirements

Python 3.12 (the published results used 3.12.11; every pinned version is
recorded at the top of `requirements.txt`). Install the dependencies with
`pip install -r requirements.txt`, then the HBV model itself, which lives in a
separate repository and is not on PyPI:

```
git clone https://github.com/sopanpatil/hbv-model
export HBV_MODEL_REPO=/path/to/hbv-model        # or put it on PYTHONPATH
```

Only `generate_states.py` and `calibrate_catchment.py` import it, because
only they run HBV. If you are verifying the paper's numbers from
`derived_data/` rather than rerunning the model, you do not need it: every
verification script and the whole synthesis run without it.

`cartopy` is optional: it adds coastlines and borders to the map panels, and
every map falls back to a plain lon/lat frame without it.

## Paths

No input path is hard-coded. Every analysis script takes its inputs and outputs
as explicit arguments, so a run states exactly which files it consumed. The
defaults that do exist let a fresh clone run unchanged: the figure scripts
default `--out` to the published figure path, and `verify_manuscript.py`,
`temperature_control.py` and `verify_text_s2.py` default `--derived` to
`derived_data`. Two environment variables cover the rest:

```
export HBV_MODEL_REPO=/path/to/hbv-model     # or put that checkout alongside
                                             # this one, which is the fallback
export CARAVAN_DEST=/path/for/raw/caravan    # default ./caravan_raw
```

## Data

Three tiers, of which the smallest and most important is carried here:

| Tier | Where it comes from |
|---|---|
| Raw Caravan (~57 GB) | Download with `download_caravan.sh`. Distributed by the Caravan authors and its extension authors; not redistributed here. |
| Daily HBV states (~7 GB) | Regenerate with `generate_states.py`. Working intermediate, not carried. |
| **Derived per-catchment products (~29 MB)** | **Tracked, in [`derived_data/`](derived_data/).** Every number in the Results and in SI Tables S1-S3 comes from these, so a clone reproduces the paper without the 57 GB download or the calibration run. |

One small input is also tracked, because it is small and pins the analysis:
`teleconnection_seasonal.csv`, the CPC indices as fetched and seasonally
averaged. `dk_record_coverage.csv` sits alongside it as provenance rather
than as an input, and no script reads it: it records the daily-record coverage
of all 308 Danish gauges that were calibrated, of which 263 pass the screen
into the analysis set.

Every column of every tracked product is defined in
[`derived_data/DATA_DICTIONARY.md`](derived_data/DATA_DICTIONARY.md), including
units, what a missing value means, and which flag selects the analysis set.

Note the licence split: the code is MIT, the data in `derived_data/` are
CC BY 4.0. See `LICENSE` and `derived_data/README.md`.

## Pipeline

Run in order. Each analysis script takes its input and output paths explicitly
(`--join`, `--signal`, `--strength-memory`, `--out`, and so on); `--help` lists
them. The two Supporting Information verification scripts instead take a
single input directory, `--derived <dir>`, defaulting to `derived_data`.

**Calibration**

| Script | Role |
|---|---|
| `download_caravan.sh` | Fetch Caravan base (which carries the CAMELS-GB and LamaH-CE sources) plus the CAMELS-DK and GRDC-Caravan extensions |
| `caravan_io.py` | Loader for the Caravan per-basin netCDF schema |
| `build_attributes.py` | Assemble `attributes.parquet` from the collections' own attribute CSVs |
| `pet_penman_monteith.py` | FAO-56 Penman-Monteith PET, computed uniformly for every country |
| `calibrate_catchment.py`, `run_calibration_batch.py` | Per-catchment SCE-UA calibration of HBV against observed discharge |
| `aggregate_calibration.py` | Collect the per-basin calibration JSONs into `calibrated_parameters_ALL.csv` (a working intermediate; only the screened table below is carried) |
| `screen_analysis_sample.py` | Add attributes, the glacier screen and the LamaH/GRDC de-duplication to `calibrated_parameters_ALL_refined.csv` as reversible flags |
| `generate_states.py` | Drive each calibrated model with its full daily forcing; save every internal store |

**Forcing and response properties**

| Script | Writes | Backs |
|---|---|---|
| `fetch_teleconnection_indices.py` | `teleconnection_seasonal.csv` | The four CPC circulation indices: NAO, EA, EAWR, SCA |
| `build_seasonal_table.py` | `seasonal_join_DJF.parquet` | Daily states collapsed to seasonal means and anomalies |
| `fit_precipitation_signal.py` | `precipitation_signal_DJF.parquet` | Section 4.1, Figure 1 |
| `response_strength_and_memory.py` | `response_strength_memory_DJF.parquet` | Store memory and gains, Section 4.2, Figure 2 |
| `response_timing.py` | `response_timing_DJF.parquet` | Timing, Section 4.2.3, Figure 3 |
| `response_properties_observed.py` | `response_observed_DJF.parquet` | All three properties on observed flow, Section 4.3, Figure 4 |
| `interannual_persistence.py` | `interannual_persistence_DJF.parquet` (not carried, see [`derived_data/README.md`](derived_data/README.md)) | Nothing in the paper. A fourth candidate property, the persistence of the signal across successive winters, evaluated and not carried forward. Kept here for provenance |

**Synthesis**

| Script | Writes | Backs |
|---|---|---|
| `nesting_screen.py` | `nesting_flags.csv`, `nesting_stem_pairs.csv` | Section 2.3 and SI Text S3 |
| `physiographic_synthesis.py` | `physiographic_{coeffs,summary,by_country}_{full,independent}.csv` | Sections 4.4 to 4.6, Figure 5, SI Tables S1-S3 |
| `temperature_control.py` | `gain_temperature_control.parquet` | SI Text S1 |
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

One ambiguity to note: `--attrs` names a different file in two scripts. In
`fit_precipitation_signal.py` it is the calibration table, which carries the gauge
coordinates; in `physiographic_synthesis.py` it is `attributes.parquet`, the
climate and physiography predictors.

The precipitation signal, which reproduces `precipitation_signal_DJF.parquet`
exactly:

```
python fit_precipitation_signal.py \
    --join  derived_data/seasonal_join_DJF.parquet \
    --attrs derived_data/calibrated_parameters_ALL_refined.csv \
    --out   precipitation_signal_DJF.parquet
```

The synthesis, which regenerates SI Tables S1 to S3:

```
python physiographic_synthesis.py \
    --join             derived_data/seasonal_join_DJF.parquet \
    --signal           derived_data/precipitation_signal_DJF.parquet \
    --strength-memory  derived_data/response_strength_memory_DJF.parquet \
    --timing           derived_data/response_timing_DJF.parquet \
    --observed         derived_data/response_observed_DJF.parquet \
    --attrs            derived_data/attributes.parquet \
    --refined          derived_data/calibrated_parameters_ALL_refined.csv \
    --nesting          derived_data/nesting_flags.csv \
    --outdir           out --nboot 1999 --tag full
```

That command reproduces `physiographic_{coeffs,summary,by_country}_full.csv`
byte for byte, and so regenerates SI Tables S1 to S3 exactly. It takes about
90 seconds. Add `--independent-only --tag independent` for the spatially
independent subset. `--nboot 1999` is the published setting; a smaller value
returns the same coefficients with coarser bootstrap p-values in proportionally
less time.

The three response properties themselves cannot be recomputed from
`derived_data/` alone: `response_strength_and_memory.py`, `response_timing.py`
and `response_properties_observed.py` read the daily HBV state series, about 7 GB,
which is not carried here because `generate_states.py` regenerates it. Their
outputs are archived instead. Figure 3 depends on the same series.

## Verifying the reported numbers

Most analysis scripts carry a `--selftest` that runs on synthetic data with a
known answer and needs no archive. All but `generate_states.py` also run
without `hbv-model`:

```
for f in fit_precipitation_signal response_strength_and_memory response_timing \
         response_properties_observed physiographic_synthesis nesting_screen \
         interannual_persistence temperature_control build_attributes \
         build_seasonal_table generate_states screen_analysis_sample; do
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

Two further scripts cover the Supporting Information text sections, and exit
non-zero the same way:

```
python temperature_control.py   # SI Text S1
python verify_text_s2.py        # SI Text S2
```

Supporting Information Tables S1 to S3 are rendered directly from
`derived_data/physiographic_*.csv`, so they are checked by construction.

## Licence and citation

Code: MIT (see `LICENSE`). Data in `derived_data/`: CC BY 4.0. If you use
either, cite the software and the paper, as `CITATION.cff` sets out.

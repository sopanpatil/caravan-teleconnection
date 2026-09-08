# derived_data

Per-catchment derived products underlying the paper. Every number in the
Results, and every entry in Supporting Information Tables S1 to S3, comes from
these files, so they are versioned here rather than left to be regenerated.

They are small enough to carry (about 29 MB, one file of 25 MB and fifteen
under 1.2 MB), and carrying them means a clone reproduces every published
number without the 57 GB raw Caravan download or the calibration run.

- **Model**: https://github.com/sopanpatil/hbv-model, doi:10.5281/zenodo.21860981.
- **Source data**: Caravan, including the CAMELS-GB and LamaH-CE base sources
  and the CAMELS-DK and GRDC-Caravan extensions, cited in the paper's Open
  Research statement. Raw
  Caravan is not redistributed here; download it with `../download_caravan.sh`.
- **Regenerating these**: see the pipeline table in the top-level README.

## Contents

| file | rows | cols | what it is |
|---|---:|---:|---|
| `calibrated_parameters_ALL_refined.csv` | 3201 | 29 | Calibrated HBV parameters for every catchment attempted, with calibration and validation KGE, HydroATLAS glacier cover, LamaH/GRDC duplicate flags, and `include_in_analysis`. The 2,135-catchment analysis set is the subset with `include_in_analysis` true. |
| `attributes.parquet` | 7195 | 13 | Caravan climate signatures and HydroATLAS physiography, for the domain before screening |
| `seasonal_join_DJF.parquet` | 135186 | 27 | DJF seasonal means and anomalies per catchment-winter, joined to the circulation indices. Restricted to the 2,135 analysis catchments. The input to every stage below, and the one file that lets the analysis be redone without the raw Caravan download. |
| `stage1_DJF.parquet` | 2135 | 33 | Per-catchment precipitation sensitivity to the four modes: partial coefficients, standard errors, p and FDR q values, adjusted R2 |
| `stage2_DJF.parquet` | 2135 | 27 | Memory (e-folding tau, lag-1 ac1, censoring flags) and standardised gains, per store, on simulated flow |
| `stage2c_DJF.parquet` | 2135 | 14 | Timing on simulated flow: response lag, late-response fraction, winter correlation, retention |
| `stage2_obs_DJF.parquet` | 2135 | 17 | All three properties recomputed on observed discharge, plus the gap-masked simulated memory used for the like-for-like comparison |
| `gain_temperature_control.parquet` | 2135 | 10 | Response strength with and without the fitted temperature signal entered alongside the precipitation forcing (SI Text S1) |
| `nesting_flags.csv` | 2135 | 6 | Geometric spatial-independence screen: equivalent radius, gauges contained, `independent` flag |
| `nesting_stem_pairs.csv` | 639 | 6 | Gauge pairs identified as consecutive on one main stem |
| `stage3_full_coeffs_{full,independent}.csv` | 60 | 7 | Standardised coefficients, country-clustered SEs, wild cluster bootstrap p and BH q, per response and predictor (SI Table S2) |
| `stage3_full_summary_{full,independent}.csv` | 10 | 23 | Marginal and within-country R2, ICC, and cross-validation skill under both references, per response (SI Table S1) |
| `stage3_full_by_country_{full,independent}.csv` | 157 | 11 | Leave-one-country-out skill per withheld country, decomposed into level offset and within-country ordering (SI Table S3) |

`_full` is the 2,135-catchment analysis set; `_independent` is the
1,364-catchment spatially independent subset.

## What can and cannot be reproduced from these files

Reproducible here, and checked to match the archive exactly: the Stage-1
precipitation sensitivities (`stage1_sensitivity.py`), the spatial-independence
screen (`nesting_screen.py`), the whole Stage-3 synthesis and both
cross-validation schemes (`stage3_full_synthesis.py`, SI Tables S1 to S3), the
temperature control (SI Text S1), the memory-retention test (SI Text S2), and
Figures 1, 2, 4 and 5.

Not reproducible from these files alone: the three response properties
themselves (`stage2_filtering.py`, `stage2c_registration_lag.py`,
`stage2_obs_validation.py`, and Figure 3). Those read the daily HBV state
series, about 7 GB, which is not archived because it regenerates from
`generate_states.py` given the calibrated parameters here and the raw Caravan
forcing. Their outputs are archived instead, as the `stage2*` files above.

## Notes

- `calibrated_parameters_ALL_refined.csv` carries two screening flags, and only
  one of them is the paper's. `include_in_analysis` is the analysis set (KGE
  > 0.5 in both calibration and validation, plus the glacier and duplicate
  screens). `used_in_analysis` is a looser upstream flag at KGE >= 0.3 that
  nothing downstream reads; it is kept only for provenance.
- Observed discharge appears here only as DJF seasonal means and anomalies
  (`flow_obs_DJF`, `flow_obs_DJF_anom`), aggregated from the daily records
  distributed by the source collections. Attribution for those records is owed
  to the contributing national agencies and to the collections cited above.
- The `country` label carries **19 distinct values, not the 16 countries the
  paper analyses**. Great Britain arrives from its sources split into
  `England` (215), `Scotland` (117), `Wales` (29) and `Great Britain` (182),
  which sum to the 543 reported. `stage3_full_synthesis.py` consolidates the
  four before any analysis. Count countries after that consolidation, not from
  this column directly.
- The `country` label is spelled `Liechtenstein` throughout. The working
  pipeline wrote `Lichtenstein`; corrected here, with no effect on any value.
- Two intermediates from the working directory are deliberately absent.
  `stage2b_DJF.parquet` was not regenerated when the KGE screen was tightened
  and still held the earlier, looser 2,423-catchment sample; the interannual
  persistence claim it supports was separately re-verified on the final 2,135.
  Products of an Atlantic Multidecadal Oscillation sub-analysis are also absent:
  the AMO is not one of the four circulation modes this paper analyses, and a
  67-winter record spans roughly one AMO cycle, so it cannot be tested here.
- Four superseded synthesis outputs (`stage3_coeffs.csv`,
  `stage3_cv_summary.csv`, `stage3_cv_by_country.csv`,
  `stage3c_timing_coeffs.csv`) are also absent. They carry pre-re-analysis
  figures: normal-reference p-values that 16 country clusters cannot support,
  and transfer skill scored only against the global mean. The
  `stage3_full_*` files supersede them.
- Daily HBV state series (about 7 GB) are not carried here; regenerate them
  with `../generate_states.py`.

## Licence

The **data in this folder** are released under Creative Commons Attribution 4.0
International (CC BY 4.0), not the MIT licence that covers the code in the rest
of the repository. Attribute them to the paper.

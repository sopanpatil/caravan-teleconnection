# Data dictionary

Every column of every file in `derived_data/`. [`README.md`](README.md) says what
each file is and how it was made; this says what is inside it, so the data can be
reused without reading the code that wrote it.

## Conventions

These hold throughout unless a table says otherwise.

- **`gauge_id`** is `<source>_<native id>`, for example `camelsdk_100006`,
  `camelsgb_54057`, `lamah_310`, `GRDC_6335020`. It is the join key in every
  file. Note the GRDC prefix is upper case and the other three are lower case,
  which is how the source collections name them.
- **`source`** is one of `camelsdk`, `camelsgb`, `lamah`, `GRDC`.
- **Units.** Fluxes (precipitation, PET, melt, discharge) are **mm/day**. Stores
  (SM, SP, UZ, LZ) are **mm**. Temperature is **degrees Celsius**. Areas are
  **km²**. Lags are **months**, memory timescales **days**.
- **`_DJF`** is a winter mean; **`winter_year` Y means December of Y-1 plus
  January and February of Y**, so the December is rolled forward into the year
  it mostly influences. This matches the CPC index seasons exactly.
- **`_anom`** is the value minus that catchment's own long-term DJF mean, taken
  over its available post-spin-up winters. The reference is per catchment, not
  pooled.
- **Missing means not estimable, never zero.** A blank or NaN says the quantity
  could not be resolved from that catchment's record: too few winters, a
  censored autocorrelation, or no detectable signal. The tables below say which
  applies. Do not fill these with zeros.
- **The circulation indices** (NAO, EA, EAWR, SCA) are the CPC standardised
  Northern Hemisphere teleconnection indices, seasonally averaged, dimensionless.
- **Simulated vs observed.** A `Qsim` or `sim` name is the calibrated HBV's flow;
  `Qobs` or `obs` is the measured discharge. Where a property appears in both
  forms, the paper reports the observed one as the check on the simulated one.

## Which rows are the paper's

`attributes.parquet` (7,195 rows) and `calibrated_parameters_ALL_refined.csv`
(3,201 rows) cover the whole downloaded domain, before screening. Everything
else is the **2,135-catchment analysis set**, which is
`calibrated_parameters_ALL_refined.csv` filtered to `include_in_analysis == True`.
Filter on that flag rather than assuming a row count.

---

## `calibrated_parameters_ALL_refined.csv` (3,201 rows)

Calibrated HBV parameters for every catchment attempted, with the screening
flags that define the analysis set. Written by `screen_analysis_sample.py`.

| column | type | definition |
|---|---|---|
| `gauge_id` | str | Catchment identifier |
| `source` | str | Originating collection |
| `calibration_kge` | float | Kling-Gupta efficiency over the calibration period (winter years 1983-2002) |
| `validation_kge` | float | KGE over the independent validation period (2003-2020). Missing where the record does not span it |
| `n_cal_obs` | int | Days of observed discharge used in calibration |
| `n_val_obs` | int | Days of observed discharge used in validation |
| `TT` | float | Snow/rain threshold temperature (°C). Bounds -2.5 to 2.5 |
| `CFMAX` | float | Degree-day melt factor (mm/°C/day). 0.5 to 10 |
| `CFR` | float | Refreezing coefficient, as a fraction of `CFMAX` (-). 0 to 0.1 |
| `CWH` | float | Liquid water holding capacity of the snowpack, as a fraction of it (-). 0 to 0.2 |
| `FC` | float | Maximum soil moisture storage (mm). 50 to 500 |
| `LP` | float | Soil moisture above which evaporation reaches its potential rate, as a fraction of `FC` (-). 0.3 to 1 |
| `BETA` | float | Shape exponent of the recharge function `(SM/FC)^BETA` (-). 1 to 6 |
| `K0` | float | Near-surface recession coefficient, applied to upper-zone storage above `UZL` (1/day). 0.05 to 0.5 |
| `K1` | float | Upper-zone recession coefficient (1/day). 0.01 to 0.3 |
| `K2` | float | Lower-zone (baseflow) recession coefficient (1/day). 0.001 to 0.15. Its reciprocal is the baseflow timescale the paper compares `tau_LZ` against |
| `UZL` | float | Upper-zone storage threshold above which `K0` acts (mm). 0 to 100 |
| `PERC` | float | Maximum percolation from upper to lower zone (mm/day). 0 to 6 |
| `MAXBAS` | float | Base length of the triangular routing function (days). 1 to 7, rounded to an integer when applied |
| `used_in_analysis` | bool | **Not the paper's screen.** A looser upstream flag at KGE >= 0.3, kept only for provenance; nothing downstream reads it |
| `gauge_lat`, `gauge_lon` | float | Gauge coordinates (degrees) |
| `area` | float | Catchment area (km²) |
| `country` | str | **19 distinct values, not 16.** Great Britain arrives split into `England`, `Scotland`, `Wales` and `Great Britain`; `physiographic_synthesis.py` consolidates them before any analysis |
| `gla_pc_sse` | float | HydroATLAS glacier cover (%) |
| `glacier_pass` | bool | True where `gla_pc_sse` is at or below the 5 % exclusion threshold |
| `is_duplicate` | bool | True where this catchment is a LamaH/GRDC duplicate that was dropped |
| `dup_partner` | str | The `gauge_id` it duplicates, where applicable |
| `include_in_analysis` | bool | **The paper's screen.** KGE > 0.5 in both calibration and validation, plus the glacier and duplicate screens. True for the 2,135 analysis catchments |

## `attributes.parquet` (7,195 rows)

Climate signatures and physiography, for the whole domain before screening.
Written by `build_attributes.py`, which copies named columns out of the
attribute CSVs the Caravan collections ship. Nothing is recomputed.

`aridity`, `moisture_index` and `seasonality` are the **ERA5-Land** variants
where a collection publishes both those and FAO Penman-Monteith ones, so that
they rest on the same forcing as the model runs.

| column | type | source column | definition |
|---|---|---|---|
| `gauge_id` | str | | Catchment identifier |
| `aridity` | float | `aridity_ERA5_LAND` | Long-term PET/P (-) |
| `moisture_index` | float | `moisture_index_ERA5_LAND` | Caravan moisture index (-) |
| `seasonality` | float | `seasonality_ERA5_LAND` | Precipitation seasonality (-) |
| `frac_snow` | float | `frac_snow` | Fraction of precipitation falling as snow (0-1). The snow-free split in SI Text S2 is `frac_snow <= 0.05` |
| `p_mean` | float | `p_mean` | Mean daily precipitation (mm/day) |
| `gwt_depth` | float | `gwt_cm_sav` | Groundwater table depth (cm). Enters the synthesis as `log1p(gwt_depth)` |
| `karst_pc` | float | `kar_pc_sse` | Karst extent (%) |
| `slope_deg` | float | `slp_dg_sav` | Terrain slope, in HydroATLAS units of degrees × 10. Enters the synthesis logged |
| `elev_m` | float | `ele_mt_sav` | Mean elevation (m) |
| `clay_pc` | float | `cly_pc_sav` | Clay fraction (%) |
| `sand_pc` | float | `snd_pc_sav` | Sand fraction (%) |
| `snowcov_pc` | float | `snw_pc_syr` | Mean annual snow cover extent (%) |

Six of these are the synthesis predictors: `aridity`, `frac_snow`, `karst_pc`,
`clay_pc`, plus logged `slope_deg` and logged `gwt_depth`. The rest are carried
for context and are not regressed on.

## `seasonal_join_DJF.parquet` (135,186 rows)

One row per catchment-winter. The input to every downstream script, and the one
file that lets the analysis be redone without the raw Caravan download. Written
by `build_seasonal_table.py`.

| column | type | definition |
|---|---|---|
| `winter_year` | int | Winter label: December of Y-1 with January and February of Y |
| `gauge_id`, `source` | str | Catchment identifier and collection |
| `precip_DJF` | float | Winter mean precipitation (mm/day) |
| `temp_DJF` | float | Winter mean air temperature (°C) |
| `pet_DJF` | float | Winter mean FAO-56 Penman-Monteith PET (mm/day) |
| `melt_DJF` | float | Winter mean simulated snowmelt (mm/day) |
| `Qsim_DJF` | float | Winter mean simulated discharge (mm/day) |
| `flow_obs_DJF` | float | Winter mean observed discharge (mm/day). Missing for 47,328 catchment-winters where the gauge record does not cover that winter |
| `SM_DJF` | float | Winter mean soil moisture store (mm) |
| `SP_DJF` | float | Winter mean snowpack store (mm) |
| `UZ_DJF` | float | Winter mean upper-zone store (mm) |
| `LZ_DJF` | float | Winter mean lower-zone store (mm) |
| `*_DJF_anom` | float | Each of the above minus that catchment's own long-term DJF mean, same units |
| `NAO_DJF`, `EA_DJF`, `EAWR_DJF`, `SCA_DJF` | float | The four CPC circulation indices, DJF means, standardised (-) |

A winter is present only if it has at least 80 of the 90 winter days finite, and
only from five years after the catchment's first forcing day, so that the
lower-zone store has left its initial condition behind.

## `precipitation_signal_DJF.parquet` (2,135 rows)

Per-catchment regression of the DJF precipitation anomaly on the four
standardised circulation indices, fitted jointly. Written by
`fit_precipitation_signal.py`. Backs Section 4.1 and Figure 1.

| column | type | definition |
|---|---|---|
| `gauge_id`, `source` | str | Catchment identifier and collection |
| `gauge_lat`, `gauge_lon` | float | Gauge coordinates (degrees) |
| `area` | float | Catchment area (km²) |
| `n_winters` | int | Winters entering the regression |
| `r2`, `adj_r2` | float | Coefficient of determination of the joint four-mode fit, raw and adjusted. The paper's "a third of variance" is the median `adj_r2` |
| `f_pvalue` | float | p-value of the joint F test on all four modes |
| `beta_<M>` | float | Partial regression coefficient for mode M, in mm/day of DJF precipitation anomaly per standard deviation of the index. M is one of `NAO`, `EA`, `EAWR`, `SCA` |
| `se_<M>` | float | Standard error of `beta_<M>` |
| `t_<M>`, `p_<M>` | float | t statistic and two-sided p-value for `beta_<M>` |
| `q_<M>` | float | Benjamini-Hochberg false-discovery-rate q-value, computed across catchments within each mode |
| `sig_<M>` | bool | True where `q_<M>` clears the FDR threshold. These are the shares quoted in Section 4.1 |

The fitted forcing the response properties are built on is
`P̂' = Σ beta_M × index_M`, reconstructed from these coefficients rather than
stored as a column.

## `response_strength_memory_DJF.parquet` (2,135 rows)

Memory and store-wise gains on simulated series. Written by
`response_strength_and_memory.py`. Backs Sections 4.2.1-4.2.2 and Figure 2.
`<S>` is one of `SM` (soil moisture), `SP` (snowpack), `UZ` (upper zone),
`LZ` (lower zone), `Qsim` (simulated flow).

| column | type | definition |
|---|---|---|
| `gauge_id`, `source` | str | Catchment identifier and collection |
| `tau_<S>` | float | **Memory.** e-folding lag (days) of the autocorrelation of the deseasonalised, detrended daily series. **Missing where censored** |
| `cens_<S>` | bool | True where the autocorrelation had not fallen to 1/e within the five-year search horizon, so `tau_<S>` is missing. Common for LZ (135 catchments), which is near-integrated at daily resolution. This is the honest statement, not a defect |
| `ac1_<S>` | float | Lag-1 autocorrelation of the same anomaly series. A bounded [0,1] memory proxy that is always defined; use it where `tau` is censored. Higher means longer memory |
| `sp_active_frac` | float | Fraction of winters with a snowpack anomaly present. **The paper's snow split is `sp_active_frac >= 0.3`**, which is "snow-active" |
| `gain_<S>` | float | **Response strength.** OLS slope of that store's DJF anomaly on the fitted forcing P̂'. Units are the store's units per mm/day of fitted precipitation anomaly. `gain_Qsim` is the winter transfer coefficient the paper reports |
| `n_winters_gain` | int | Winters entering the gain regression. Gains need at least 20 |
| `gauge_lat`, `gauge_lon`, `area` | float | Coordinates (degrees) and area (km²) |

Expect `tau_LZ` >> `tau_SM` > `tau_UZ`. `tau_SP` is meaningful only where
`sp_active_frac` is appreciable.

## `response_timing_DJF.parquet` (2,135 rows)

When the catchment releases the winter signal across the water year, on
simulated flow. Written by `response_timing.py`. Backs Section 4.2.3 and
Figure 3.

The forcing stays strictly winter; the response is tracked at lags L = 0 to 11
months after the December (L=0 December, 1 January, ... 7 July). The
noise-corrected energy profile is `w(L) = max(r(L)² - 1/(n-2), 0)`.

| column | type | definition |
|---|---|---|
| `gauge_id`, `source` | str | Catchment identifier and collection |
| `reg_lag` | float | **Timing.** Energy-weighted centroid lag (months). Sign-agnostic, so a melt registration counts whatever its sign. Flashy catchments sit near 0-2, snow-dominated ones near 4+ |
| `late_frac` | float | Fraction of response energy at L >= 4 (April onward). A bounded [0,1] phase-shift index: near 0 flashy, high for snow. This is the timing variable used in the transfer test |
| `winter_r` | float | Correlation at the winter lags L = 0-2, the contemporaneous registration strength |
| `peak_lag` | float | Lag (months) at which \|r\| peaks |
| `sig` | float | Peak \|r\| over the window: signal strength. **The paper gates timing comparisons on `sig > 0.2`**, below which the lag is ill-defined |
| `retention` | float | Share of the early (L = 0-6) response energy arriving in March or later, `sum(w[3:7]) / sum(w[0:7])`. How long the forced anomaly lasts, as opposed to when it arrives. This is the SI Text S2 test statistic. Missing where the early window carries no energy |
| `n_lags` | int | Lags with a finite correlation |
| `tau_Qsim`, `ac1_Qsim`, `sp_active_frac` | float | Carried through from the memory file for convenience |
| `gauge_lat`, `gauge_lon` | float | Gauge coordinates (degrees) |

## `response_observed_DJF.parquet` (2,135 rows)

All three properties recomputed on measured discharge, plus the gap-masked
simulated memory that makes the memory comparison like-for-like. Written by
`response_properties_observed.py`. Backs Section 4.3 and Figure 4.

| column | type | definition |
|---|---|---|
| `gauge_id`, `source` | str | Catchment identifier and collection |
| `tau_Qobs` | float | Memory of observed flow (days), same estimator as `tau_Qsim` |
| `cens_Qobs` | bool | True where that e-folding was censored |
| `ac1_Qobs` | float | Lag-1 autocorrelation of observed flow |
| `ndays_Qobs` | int | Valid observed days used |
| `tau_Qsim_m`, `cens_Qsim_m`, `ac1_Qsim_m`, `ndays_Qsim_m` | | **The `_m` suffix means gap-masked**: the simulated series with the observed record's gaps punched into it, so the two memories are estimated on identical days. The paper's simulated-vs-observed memory correlation uses these, not the ungapped `tau_Qsim` |
| `reg_lag_obs`, `late_frac_obs`, `winter_r_obs`, `peak_lag_obs`, `sig_obs`, `retention_obs` | float | The timing quantities above, recomputed on observed flow. Same definitions, same units. Missing for the ~92 catchments where the observed profile could not be formed |
| `n_lags_obs` | int | Lags with a finite correlation on observed flow |

## `gain_temperature_control.parquet` (2,135 rows)

The exclusion-restriction check behind the response strength: does controlling
for the temperature pathway change the answer? Written by
`temperature_control.py`. Backs SI Text S1.

| column | type | definition |
|---|---|---|
| `gauge_id` | str | Catchment identifier |
| `r_phat_that` | float | Correlation between the fitted precipitation signal P̂' and the analogously fitted temperature signal T̂'. Higher means the two pathways are more entangled |
| `gain_obs`, `gain_sim` | float | Response strength on observed and simulated flow, single-regressor, P̂' only. Missing on observed flow for 116 catchments |
| `gain_obs_tctrl`, `gain_sim_tctrl` | float | The same slopes with T̂' entered alongside P̂'. **`_tctrl` means temperature-controlled** |
| `n_winters_obs`, `n_winters_sim` | int | Winters entering each single-regressor fit |
| `n_winters_obs_tctrl`, `n_winters_sim_tctrl` | int | Winters entering each controlled fit |

The paper reports that the paired change is small and the snow contrast
survives, so the single-regressor coefficient is used throughout.

## `nesting_flags.csv` (2,135 rows)

Geometric spatial-independence screen. Written by `nesting_screen.py`. Backs
Section 2.3 and SI Text S3.

| column | type | definition |
|---|---|---|
| `gauge_id` | str | Catchment identifier |
| `area` | float | Catchment area (km²) |
| `equiv_radius_km` | float | Radius of a disc of that area, `sqrt(area/pi)`, in km |
| `n_gauges_inside` | int | Retained gauges falling within that radius with a smaller area |
| `is_container` | bool | True where this catchment geometrically contains another retained gauge |
| `independent` | bool | **The subset flag.** True for the 1,364 catchments in the spatially independent subsample. Filter on this to reproduce every `_independent` result |

## `nesting_stem_pairs.csv` (639 rows)

Gauge pairs the screen identified as consecutive on one main stem. Reported in
SI Text S3; not used to filter anything.

| column | type | definition |
|---|---|---|
| `gauge_a`, `gauge_b` | str | The two gauges, `a` the larger catchment |
| `area_a`, `area_b` | float | Their areas (km²) |
| `dist_km` | float | Great-circle distance between the gauges (km) |
| `area_ratio` | float | `area_b / area_a`, so near 1 means the pair gauges nearly the same water |

## `physiographic_coeffs_{full,independent}.csv` (60 rows)

Standardised regression coefficients, one row per response × predictor. Written
by `physiographic_synthesis.py`. **This is SI Table S2.** `_full` is the 2,135
analysis set, `_independent` the 1,364 spatially independent subset.

| column | type | definition |
|---|---|---|
| `response` | str | One of the ten responses listed below |
| `predictor` | str | One of `aridity`, `frac_snow`, `slope_l`, `clay_pc`, `gwt_l`, `karst_pc`. The `_l` suffix marks the logged form |
| `beta_std` | float | Standardised coefficient: response standard deviations per predictor standard deviation |
| `se_clustered` | float | Country-clustered robust standard error |
| `t` | float | `beta_std / se_clustered` |
| `p_wild_bootstrap` | float | **The paper's p-value.** Restricted wild cluster bootstrap, Rademacher weights, 1,999 replicates, resampled by country. Not a monotone function of `t`, deliberately: see SI Text S4 |
| `q_bh` | float | Benjamini-Hochberg q-value over the bootstrap p-values |

Use `p_wild_bootstrap`, not a normal reference on `t`. With 16 country clusters
the cluster-robust sandwich is badly downward-biased, which is the whole reason
the bootstrap is there.

## `physiographic_summary_{full,independent}.csv` (10 rows)

One row per response: explanatory power, clustering, and out-of-sample skill.
**This is SI Table S1.**

| column | type | definition |
|---|---|---|
| `response` | str | Response name |
| `label` | str | Human-readable description |
| `family` | str | `amplitude`, `memory` or `timing` |
| `n` | int | Catchments entering the fit |
| `n_countries` | int | Countries represented |
| `marginal_R2` | float | R² of the six standardised predictors alone |
| `country_partial_R2` | float | R² added by country indicators over the predictors |
| `predictor_partial_R2_within` | float | R² the predictors add over country indicators alone, that is, their within-country explanatory power |
| `icc` | float | Residual intraclass correlation by country, method of moments. High means the residual is mostly a national level offset |
| `kfold_r2_global` | float | Ten-fold cross-validated R², folds drawn at random, scored against the global mean |
| `loco_r2_global` | float | **Leave-one-country-out** R², scored against the global mean. Flatters the model, because it gets credit for placing a country at roughly the right level |
| `loco_r2_local_pooled` | float | The same predictions scored against each withheld **country's own mean**. This is the honest transfer skill, and it is what the paper leads with |
| `loco_r2_local_debiased_pooled` | float | As above after removing each country's mean prediction error, so it isolates whether the within-country **ordering** is right |
| `loco_r2_local_median_country` | float | Median of the per-country local R² |
| `loco_r2_local_wmean_country` | float | Catchment-weighted mean of the per-country local R². **Entirely missing in both files**; use the median or the pooled figures |
| `loco_n_countries_negative_local` | int | Countries scoring below their own mean |
| `loco_n_countries_big` | int | Countries with at least 10 catchments, the subset the summaries below restrict to |
| `loco_r2_local_median_big` | float | Median local R² over those countries |
| `loco_r2_local_debiased_median_big` | float | Median de-biased local R² over those countries |
| `loco_rho_within_median_big` | float | Median within-country Spearman correlation between prediction and truth |
| `loco_n_big_negative_local` | int | Of those countries, how many score below their own mean |
| `loco_n_big_negative_debiased` | int | How many still do after de-biasing |
| `loco_n_big_negative_rho` | int | How many order no better than at random |

## `physiographic_by_country_{full,independent}.csv` (157 rows)

Leave-one-country-out skill for each withheld country, one row per response ×
country. **This is SI Table S3.**

| column | type | definition |
|---|---|---|
| `response`, `country` | str | Response and the withheld country |
| `n` | int | Catchments in that country |
| `r2_global` | float | R² for that country's predictions, scored against the global mean |
| `r2_local` | float | Scored against that country's own mean. Missing where the country has too few catchments for a stable local variance |
| `r2_local_debiased` | float | As above after removing the country's mean prediction error: the ordering component |
| `ss_local` | float | Total sum of squares about the country's own mean |
| `sse` | float | Sum of squared prediction errors |
| `sse_debiased` | float | Sum of squared errors after removing the mean bias |
| `rho_within` | float | Within-country Spearman correlation between prediction and truth |
| `mean_bias` | float | Mean prediction minus mean truth: the national level offset. Great Britain and Denmark carry the largest ones |

## The ten responses

Used as the `response` value in all three synthesis files. Each is fitted on
simulated flow and, where estimable, on observed flow, so a model-derived result
can be checked against the data that constrained it.

| response | family | what it is |
|---|---|---|
| `gain_Qsim` | amplitude | Winter transfer coefficient, simulated |
| `gain_Qobs` | amplitude | Winter transfer coefficient, observed |
| `log_tau` | memory | log of the memory timescale, simulated |
| `logit_ac1` | memory | logit of lag-1 autocorrelation, simulated |
| `log_tau_obs` | memory | log memory timescale, observed |
| `logit_ac1_obs` | memory | logit lag-1 autocorrelation, observed |
| `reg_lag` | timing | Response lag in months, simulated |
| `logit_late` | timing | logit of the late-response fraction, simulated |
| `reg_lag_obs` | timing | Response lag, observed |
| `logit_late_obs` | timing | logit late fraction, observed |

Memory and timing are transformed (log, logit) because the raw metrics are
bounded or span orders of magnitude; the amplitude is not.

# tsfresh Feature Relevance Summary

Selection uses only train bearings (`Bearing1_1`, `Bearing1_2`, `Bearing1_4`, `Bearing1_5`) and keeps `Bearing1_3` held out for RUL baselines.

This run compares `MinimalFCParameters` and `EfficientFCParameters` on single-snapshot vibration features. The conclusion is intentionally conservative: 相关性整体偏弱, so tsfresh is treated as an evaluated automatic feature candidate rather than a core performance breakthrough.

## Configuration Overview

- `EfficientFCParameters`: 1554 generated features in this summary, 32 selected features, top train-only correlation `0.424468`.
- `MinimalFCParameters`: 20 generated features in this summary, 8 selected features, top train-only correlation `0.200994`.

## Top Correlation Features

| feature_set_config | feature_name | score | p_value | correlation | selected | feature_group | overlaps_manual_19 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| EfficientFCParameters | horizontal__partial_autocorrelation__lag_1 | 0.424468 | 2.77475e-21 | 0.424468 | True | time_domain | False |
| EfficientFCParameters | horizontal__autocorrelation__lag_1 | 0.424468 | 2.77475e-21 | 0.424468 | True | time_domain | False |
| EfficientFCParameters | horizontal__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0 | 0.421501 | 5.57479e-21 | -0.421501 | True | time_domain | False |
| EfficientFCParameters | horizontal__cid_ce__normalize_True | 0.415251 | 2.3702e-20 | -0.415251 | True | time_domain | False |
| EfficientFCParameters | horizontal__cid_ce__normalize_False | 0.415251 | 2.3702e-20 | -0.415251 | True | time_domain | False |
| EfficientFCParameters | horizontal__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0 | 0.413182 | 3.80199e-20 | -0.413182 | True | time_domain | True |
| EfficientFCParameters | horizontal__mean_abs_change | 0.413182 | 3.80199e-20 | -0.413182 | True | time_domain | True |
| EfficientFCParameters | horizontal__absolute_sum_of_changes | 0.413182 | 3.80199e-20 | -0.413182 | True | energy | False |
| EfficientFCParameters | vertical__ar_coefficient__coeff_8__k_10 | 0.412837 | 4.11205e-20 | 0.412837 | True | time_domain | False |
| EfficientFCParameters | horizontal__number_crossing_m__m_0 | 0.403452 | 3.35785e-19 | -0.403452 | True | time_domain | False |
| MinimalFCParameters | vertical__maximum | 0.200994 | 1.59646e-05 | 0.200994 | True | time_domain | False |
| MinimalFCParameters | vertical__absolute_maximum | 0.125944 | 0.00721317 | 0.125944 | True | energy | False |
| MinimalFCParameters | horizontal__maximum | 0.041937 | 0.372668 | 0.041937 | True | time_domain | False |
| MinimalFCParameters | horizontal__minimum | 0.040404 | 0.390412 | 0.040404 | True | time_domain | False |
| MinimalFCParameters | vertical__minimum | 0.035982 | 0.444387 | -0.035982 | True | time_domain | False |
| MinimalFCParameters | horizontal__absolute_maximum | 0.035392 | 0.451893 | 0.035392 | True | energy | False |
| MinimalFCParameters | vertical__median | 0.031177 | 0.507565 | -0.031177 | True | time_domain | False |
| MinimalFCParameters | horizontal__median | 0.027219 | 0.562942 | -0.027219 | True | time_domain | False |
| MinimalFCParameters | vertical__length | 0.000000 | 1 | 0.000000 | False | shape | False |
| MinimalFCParameters | vertical__root_mean_square | 0.000000 | 1 | 0.000000 | False | energy | True |

All feature scores are correlation-derived screening scores on train rows. The downstream baseline transforms held-out rows with the already selected feature list only.

Generated figures: `tsfresh_feature_correlation_bar.png`, `tsfresh_feature_group_distribution.png`, and `tsfresh_top_feature_rul_trend.png`.

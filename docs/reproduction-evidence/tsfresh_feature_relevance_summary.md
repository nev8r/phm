# tsfresh Feature Relevance Summary

Selection uses only train bearings (`Bearing1_1`, `Bearing1_2`, `Bearing1_4`, `Bearing1_5`) and keeps `Bearing1_3` held out for RUL baselines.

| feature_name | score | p_value | correlation | selected | feature_group | overlaps_manual_19 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| vertical__maximum | 0.200994 | 1.59646e-05 | 0.200994 | True | time_domain | False |
| vertical__absolute_maximum | 0.125944 | 0.00721317 | 0.125944 | True | energy | False |
| horizontal__maximum | 0.041937 | 0.372668 | 0.041937 | True | time_domain | False |
| horizontal__minimum | 0.040404 | 0.390412 | 0.040404 | True | time_domain | False |
| vertical__minimum | 0.035982 | 0.444387 | -0.035982 | True | time_domain | False |
| horizontal__absolute_maximum | 0.035392 | 0.451893 | 0.035392 | True | energy | False |
| vertical__median | 0.031177 | 0.507565 | -0.031177 | True | time_domain | False |
| horizontal__median | 0.027219 | 0.562942 | -0.027219 | True | time_domain | False |
| vertical__length | 0.000000 | 1 | 0.000000 | False | shape | False |
| vertical__root_mean_square | 0.000000 | 1 | 0.000000 | False | energy | True |
| vertical__variance | 0.000000 | 1 | 0.000000 | False | distribution | True |
| vertical__standard_deviation | 0.000000 | 1 | 0.000000 | False | distribution | True |
| horizontal__sum_values | 0.000000 | 1 | 0.000000 | False | time_domain | False |
| vertical__mean | 0.000000 | 1 | 0.000000 | False | time_domain | True |
| horizontal__root_mean_square | 0.000000 | 1 | 0.000000 | False | energy | True |
| horizontal__variance | 0.000000 | 1 | 0.000000 | False | distribution | True |
| horizontal__standard_deviation | 0.000000 | 1 | 0.000000 | False | distribution | True |
| horizontal__length | 0.000000 | 1 | 0.000000 | False | shape | False |
| horizontal__mean | 0.000000 | 1 | 0.000000 | False | time_domain | True |
| vertical__sum_values | 0.000000 | 1 | 0.000000 | False | time_domain | False |

All feature scores are correlation-derived screening scores on train rows. The downstream baseline transforms held-out rows with the already selected feature list only.

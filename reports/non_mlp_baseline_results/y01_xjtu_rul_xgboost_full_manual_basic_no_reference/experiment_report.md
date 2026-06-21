# y01_xjtu_rul_xgboost_full_manual_basic_no_reference

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `rul_tabular`
- Model: `xgboost_regressor`
- Fit status: `completed`
- Feature count: 44
- Target columns: `piecewise_rul_norm`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `RMSE` (lower_is_better)
- Val primary: 0.302017
- Test primary: 0.431454

| Split | Metric | Value |
|---|---|---:|
| val | MAE | 0.244600 |
| val | MSE | 0.091214 |
| val | RMSE | 0.302017 |
| test | MAE | 0.373355 |
| test | MSE | 0.186152 |
| test | RMSE | 0.431454 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__rms` | 0.281477 |
| 2 | `h__time__mean_abs` | 0.279107 |
| 3 | `h__time__std` | 0.179867 |
| 4 | `mag__time__mean` | 0.045368 |
| 5 | `v__time__mean_abs` | 0.030070 |
| 6 | `v__spectral__rms_frequency` | 0.021658 |
| 7 | `mag__time__mean_abs` | 0.017926 |
| 8 | `h__time__ptp` | 0.015014 |
| 9 | `mag__spectral__bandwidth` | 0.012856 |
| 10 | `v__spectral__bandwidth` | 0.009936 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

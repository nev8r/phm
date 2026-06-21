# y08_phm_rul_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `rul_tabular`
- Model: `random_forest_regressor`
- Fit status: `completed`
- Feature count: 7
- Target columns: `piecewise_rul_norm`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `RMSE` (lower_is_better)
- Val primary: 0.292971
- Test primary: 0.337575

| Split | Metric | Value |
|---|---|---:|
| val | MAE | 0.224358 |
| val | MSE | 0.085832 |
| val | RMSE | 0.292971 |
| test | MAE | 0.237797 |
| test | MSE | 0.113957 |
| test | RMSE | 0.337575 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__std` | 0.441142 |
| 2 | `h__time__rms` | 0.195210 |
| 3 | `v__time__mean_abs` | 0.134170 |
| 4 | `h__time__mean_abs` | 0.104160 |
| 5 | `mag__time__std` | 0.060776 |
| 6 | `mag__time__mean` | 0.032640 |
| 7 | `mag__time__mean_abs` | 0.031903 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

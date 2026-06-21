# y09_phm_health_xgboost_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `health_state_tabular`
- Model: `xgboost_classifier`
- Fit status: `completed`
- Feature count: 5
- Target columns: `health_state_id`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Val primary: 0.240540
- Test primary: 0.295374

| Split | Metric | Value |
|---|---|---:|
| val | Accuracy | 0.252425 |
| val | MacroF1 | 0.262161 |
| val | WeightedF1 | 0.240540 |
| test | Accuracy | 0.285605 |
| test | MacroF1 | 0.239511 |
| test | WeightedF1 | 0.295374 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__mean_abs` | 0.228388 |
| 2 | `h__time__std` | 0.222411 |
| 3 | `mag__time__mean_abs` | 0.188705 |
| 4 | `mag__time__mean` | 0.183699 |
| 5 | `h__time__rms` | 0.176797 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

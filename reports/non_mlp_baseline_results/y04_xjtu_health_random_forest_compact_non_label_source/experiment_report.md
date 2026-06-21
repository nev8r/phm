# y04_xjtu_health_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `health_state_tabular`
- Model: `random_forest_classifier`
- Fit status: `completed`
- Feature count: 6
- Target columns: `health_state_id`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Val primary: 0.583314
- Test primary: 0.350722

| Split | Metric | Value |
|---|---|---:|
| val | Accuracy | 0.686718 |
| val | MacroF1 | 0.358017 |
| val | WeightedF1 | 0.583314 |
| test | Accuracy | 0.366337 |
| test | MacroF1 | 0.294849 |
| test | WeightedF1 | 0.350722 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__std` | 0.191895 |
| 2 | `h__time__rms` | 0.189906 |
| 3 | `mag__time__mean` | 0.157519 |
| 4 | `mag__time__std` | 0.156938 |
| 5 | `h__time__mean_abs` | 0.152538 |
| 6 | `mag__time__mean_abs` | 0.151204 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

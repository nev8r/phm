# y06_xjtu_early_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `early_fault_tabular`
- Model: `random_forest_classifier`
- Fit status: `completed`
- Feature count: 5
- Target columns: `early_fault`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Val primary: 0.625932
- Test primary: 0.837047

| Split | Metric | Value |
|---|---|---:|
| val | Accuracy | 0.695057 |
| val | MacroF1 | 0.541814 |
| val | WeightedF1 | 0.625932 |
| test | Accuracy | 0.847525 |
| test | MacroF1 | 0.821127 |
| test | WeightedF1 | 0.837047 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `mag__time__mean_abs` | 0.314616 |
| 2 | `mag__time__mean` | 0.314452 |
| 3 | `v__time__mean_abs` | 0.148789 |
| 4 | `v__time__std` | 0.117221 |
| 5 | `mag__time__std` | 0.104922 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

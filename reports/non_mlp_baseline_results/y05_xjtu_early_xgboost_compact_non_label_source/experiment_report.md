# y05_xjtu_early_xgboost_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `early_fault_tabular`
- Model: `xgboost_classifier`
- Fit status: `completed`
- Feature count: 5
- Target columns: `early_fault`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Val primary: 0.630984
- Test primary: 0.839369

| Split | Metric | Value |
|---|---|---:|
| val | Accuracy | 0.708160 |
| val | MacroF1 | 0.544203 |
| val | WeightedF1 | 0.630984 |
| test | Accuracy | 0.849505 |
| test | MacroF1 | 0.823764 |
| test | WeightedF1 | 0.839369 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `mag__time__mean` | 0.396931 |
| 2 | `mag__time__mean_abs` | 0.389862 |
| 3 | `mag__time__std` | 0.084951 |
| 4 | `v__time__std` | 0.080242 |
| 5 | `v__time__mean_abs` | 0.048014 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

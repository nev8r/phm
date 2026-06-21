# y03_xjtu_health_xgboost_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `health_state_tabular`
- Model: `xgboost_classifier`
- Fit status: `completed`
- Feature count: 6
- Target columns: `health_state_id`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Train primary: 0.924950
- Val primary: 0.583939
- Test primary: 0.365121
- Train-val gap: 0.341011
- Val-test gap: 0.218818
- Gap pattern: `train_best_test_worst`

| Split | Metric | Value |
|---|---|---:|
| train | Accuracy | 0.926052 |
| train | MacroF1 | 0.852856 |
| train | WeightedF1 | 0.924950 |
| val | Accuracy | 0.690292 |
| val | MacroF1 | 0.358526 |
| val | WeightedF1 | 0.583939 |
| test | Accuracy | 0.378218 |
| test | MacroF1 | 0.309833 |
| test | WeightedF1 | 0.365121 |

## Training Adequacy

WeightedF1 is best on train and degrades on validation/test; this suggests overfitting or split distribution shift rather than too few fit iterations.

## Visual Checks

| File | Purpose |
|---|---|
| `figures/train_confusion_matrix.png` | prediction quality / class behavior |
| `figures/val_confusion_matrix.png` | prediction quality / class behavior |
| `figures/test_confusion_matrix.png` | prediction quality / class behavior |
| `figures/test_class_distribution.png` | prediction quality / class behavior |
| `figures/feature_importance_top10.png` | feature importance |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__std` | 0.262428 |
| 2 | `h__time__rms` | 0.258435 |
| 3 | `h__time__mean_abs` | 0.249001 |
| 4 | `mag__time__mean` | 0.089585 |
| 5 | `mag__time__mean_abs` | 0.084279 |
| 6 | `mag__time__std` | 0.056273 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

# y11_phm_early_xgboost_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `early_fault_tabular`
- Model: `xgboost_classifier`
- Fit status: `completed`
- Feature count: 7
- Target columns: `early_fault`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Train primary: 0.926569
- Val primary: 0.491115
- Test primary: 0.578165
- Train-val gap: 0.435453
- Val-test gap: -0.087050
- Gap pattern: `no_clear_generalization_gap`

| Split | Metric | Value |
|---|---|---:|
| train | Accuracy | 0.926998 |
| train | MacroF1 | 0.919946 |
| train | WeightedF1 | 0.926569 |
| val | Accuracy | 0.484527 |
| val | MacroF1 | 0.465642 |
| val | WeightedF1 | 0.491115 |
| test | Accuracy | 0.603301 |
| test | MacroF1 | 0.577239 |
| test | WeightedF1 | 0.578165 |

## Training Adequacy

WeightedF1 does not show a clear train-to-heldout degradation pattern; remaining error is more likely task difficulty or feature capacity.

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
| 1 | `v__time__mean_abs` | 0.266352 |
| 2 | `h__time__mean_abs` | 0.216209 |
| 3 | `v__time__std` | 0.210845 |
| 4 | `mag__time__mean` | 0.129615 |
| 5 | `mag__time__mean_abs` | 0.059845 |
| 6 | `h__time__std` | 0.059717 |
| 7 | `h__time__rms` | 0.057418 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

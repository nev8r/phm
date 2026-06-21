# y10_phm_health_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `health_state_tabular`
- Model: `random_forest_classifier`
- Fit status: `completed`
- Feature count: 5
- Target columns: `health_state_id`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Train primary: 0.963468
- Val primary: 0.261532
- Test primary: 0.340471
- Train-val gap: 0.701935
- Val-test gap: -0.078938
- Gap pattern: `no_clear_generalization_gap`

| Split | Metric | Value |
|---|---|---:|
| train | Accuracy | 0.963499 |
| train | MacroF1 | 0.965290 |
| train | WeightedF1 | 0.963468 |
| val | Accuracy | 0.278984 |
| val | MacroF1 | 0.282735 |
| val | WeightedF1 | 0.261532 |
| test | Accuracy | 0.333512 |
| test | MacroF1 | 0.269367 |
| test | WeightedF1 | 0.340471 |

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
| 1 | `h__time__mean_abs` | 0.231337 |
| 2 | `h__time__std` | 0.203141 |
| 3 | `h__time__rms` | 0.189916 |
| 4 | `mag__time__mean` | 0.188898 |
| 5 | `mag__time__mean_abs` | 0.186709 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

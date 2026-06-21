# y02_xjtu_rul_random_forest_full_manual_basic_no_reference

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `XJTU-SY`
- Split: `xjtu_bearing_index_split`
- Task: `rul_tabular`
- Model: `random_forest_regressor`
- Fit status: `completed`
- Feature count: 44
- Target columns: `piecewise_rul_norm`
- Train / Val / Test examples: 7032 / 1679 / 505

## Metrics

- Primary metric: `RMSE` (lower_is_better)
- Train primary: 0.024701
- Val primary: 0.283495
- Test primary: 0.399658
- Train-val gap: 0.258795
- Val-test gap: 0.116163
- Gap pattern: `train_best_test_worst`

| Split | Metric | Value |
|---|---|---:|
| train | MAE | 0.008015 |
| train | MSE | 0.000610 |
| train | RMSE | 0.024701 |
| val | MAE | 0.170987 |
| val | MSE | 0.080370 |
| val | RMSE | 0.283495 |
| test | MAE | 0.315260 |
| test | MSE | 0.159726 |
| test | RMSE | 0.399658 |

## Training Adequacy

RMSE is best on train and degrades on validation/test; this suggests overfitting or split distribution shift rather than too few fit iterations.

## Visual Checks

| File | Purpose |
|---|---|
| `figures/train_pred_vs_true.png` | prediction quality / class behavior |
| `figures/val_pred_vs_true.png` | prediction quality / class behavior |
| `figures/test_pred_vs_true.png` | prediction quality / class behavior |
| `figures/test_residuals.png` | prediction quality / class behavior |
| `figures/feature_importance_top10.png` | feature importance |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__mean_abs` | 0.616236 |
| 2 | `h__time__rms` | 0.153194 |
| 3 | `h__time__std` | 0.101821 |
| 4 | `h__spectral__bandwidth` | 0.014700 |
| 5 | `mag__spectral__centroid` | 0.012343 |
| 6 | `h__spectral__entropy` | 0.009573 |
| 7 | `v__time__kurtosis` | 0.009356 |
| 8 | `v__time__skewness` | 0.008743 |
| 9 | `v__spectral__rms_frequency` | 0.007273 |
| 10 | `mag__spectral__entropy` | 0.006382 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

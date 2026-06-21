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
- Train primary: 0.046758
- Val primary: 0.302017
- Test primary: 0.431454
- Train-val gap: 0.255259
- Val-test gap: 0.129437
- Gap pattern: `train_best_test_worst`

| Split | Metric | Value |
|---|---|---:|
| train | MAE | 0.020947 |
| train | MSE | 0.002186 |
| train | RMSE | 0.046758 |
| val | MAE | 0.244600 |
| val | MSE | 0.091214 |
| val | RMSE | 0.302017 |
| test | MAE | 0.373355 |
| test | MSE | 0.186152 |
| test | RMSE | 0.431454 |

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

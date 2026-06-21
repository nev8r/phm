# y07_phm_rul_xgboost_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `rul_tabular`
- Model: `xgboost_regressor`
- Fit status: `completed`
- Feature count: 7
- Target columns: `piecewise_rul_norm`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `RMSE` (lower_is_better)
- Train primary: 0.145711
- Val primary: 0.291317
- Test primary: 0.357105
- Train-val gap: 0.145606
- Val-test gap: 0.065788
- Gap pattern: `train_best_test_worst`

| Split | Metric | Value |
|---|---|---:|
| train | MAE | 0.092094 |
| train | MSE | 0.021232 |
| train | RMSE | 0.145711 |
| val | MAE | 0.234196 |
| val | MSE | 0.084865 |
| val | RMSE | 0.291317 |
| test | MAE | 0.269212 |
| test | MSE | 0.127524 |
| test | RMSE | 0.357105 |

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
| 1 | `h__time__rms` | 0.526424 |
| 2 | `h__time__std` | 0.263978 |
| 3 | `v__time__mean_abs` | 0.066934 |
| 4 | `h__time__mean_abs` | 0.050798 |
| 5 | `mag__time__mean_abs` | 0.041408 |
| 6 | `mag__time__mean` | 0.029828 |
| 7 | `mag__time__std` | 0.020630 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

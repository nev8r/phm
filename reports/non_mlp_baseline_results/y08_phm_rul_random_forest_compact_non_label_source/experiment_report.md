# y08_phm_rul_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `rul_tabular`
- Model: `random_forest_regressor`
- Fit status: `completed`
- Feature count: 7
- Target columns: `piecewise_rul_norm`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `RMSE` (lower_is_better)
- Train primary: 0.065260
- Val primary: 0.292971
- Test primary: 0.337575
- Train-val gap: 0.227711
- Val-test gap: 0.044604
- Gap pattern: `train_best_test_worse`

| Split | Metric | Value |
|---|---|---:|
| train | MAE | 0.031601 |
| train | MSE | 0.004259 |
| train | RMSE | 0.065260 |
| val | MAE | 0.224358 |
| val | MSE | 0.085832 |
| val | RMSE | 0.292971 |
| test | MAE | 0.237797 |
| test | MSE | 0.113957 |
| test | RMSE | 0.337575 |

## Training Adequacy

RMSE is best on train and test is mildly worse than validation; held-out behavior should be inspected with the generated plots.

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
| 1 | `h__time__std` | 0.441142 |
| 2 | `h__time__rms` | 0.195210 |
| 3 | `v__time__mean_abs` | 0.134170 |
| 4 | `h__time__mean_abs` | 0.104160 |
| 5 | `mag__time__std` | 0.060776 |
| 6 | `mag__time__mean` | 0.032640 |
| 7 | `mag__time__mean_abs` | 0.031903 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

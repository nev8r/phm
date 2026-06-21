# Step Y: XGBoost and RandomForest Non-MLP Tabular Baselines

## 1. Purpose

Step Y runs real non-MLP tabular baselines with XGBoost and RandomForest. It is not a dry run and it does not use the torch trainer. The recipe reuses the same dataset roots, splits, `manual_basic` feature extraction, `degradation_three_tasks` labels, and `TaskBuilder` feature selection used by the MLP baseline cycle.

These models do not train by MLP-style epochs. XGBoost uses `n_estimators=300` boosting rounds/trees, and RandomForest uses `n_estimators=300` trees. Runtime is therefore dominated by rebuilding manual features from raw CSV files plus tree fitting, not by epoch logs.

## 2. Repository Context

- Main `mode=train` remains torch-based and unchanged.
- Step Y uses `recipes/baselines/run_sklearn_baseline.py` as a standalone recipe.
- Raw model pickle and prediction parquet files are written under `artifacts/non_mlp_baselines/` and are not intended for Git commits.
- Curated reports only contain small text, JSON, and CSV outputs under `reports/non_mlp_baseline_results/`.

## 3. Models

| Model | Task Type | Fixed Parameters |
|---|---|---|
| XGBRegressor | RUL regression | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `objective=reg:squarederror`, `tree_method=hist`, `n_jobs=-1`, `random_state=42` |
| XGBClassifier | HealthState / EarlyFault classification | Same tree settings; HealthState uses `objective=multi:softprob`, `eval_metric=mlogloss`; EarlyFault uses `objective=binary:logistic`, `eval_metric=logloss` |
| RandomForestRegressor | RUL regression | `n_estimators=300`, `min_samples_leaf=2`, `n_jobs=-1`, `random_state=42` |
| RandomForestClassifier | HealthState / EarlyFault classification | `n_estimators=300`, `min_samples_leaf=2`, `class_weight=balanced`, `n_jobs=-1`, `random_state=42` |

## 4. Experiments

| ID | Dataset | Task | Model | Feature Subset | Feature Count | Target | Status |
|---|---|---|---|---|---:|---|---|
| Y1 | XJTU-SY | RUL | XGBoost | `full_manual_basic_no_reference` | 44 | `piecewise_rul_norm` | completed |
| Y2 | XJTU-SY | RUL | RandomForest | `full_manual_basic_no_reference` | 44 | `piecewise_rul_norm` | completed |
| Y3 | XJTU-SY | HealthState | XGBoost | `compact_non_label_source` | 6 | `health_state_id` | completed |
| Y4 | XJTU-SY | HealthState | RandomForest | `compact_non_label_source` | 6 | `health_state_id` | completed |
| Y5 | XJTU-SY | EarlyFault | XGBoost | `compact_non_label_source` | 5 | `early_fault` | completed |
| Y6 | XJTU-SY | EarlyFault | RandomForest | `compact_non_label_source` | 5 | `early_fault` | completed |
| Y7 | PHM2012 | RUL | XGBoost | `compact_non_label_source` | 7 | `piecewise_rul_norm` | completed |
| Y8 | PHM2012 | RUL | RandomForest | `compact_non_label_source` | 7 | `piecewise_rul_norm` | completed |
| Y9 | PHM2012 | HealthState | XGBoost | `compact_non_label_source` | 5 | `health_state_id` | completed |
| Y10 | PHM2012 | HealthState | RandomForest | `compact_non_label_source` | 5 | `health_state_id` | completed |
| Y11 | PHM2012 | EarlyFault | XGBoost | `compact_non_label_source` | 7 | `early_fault` | completed |
| Y12 | PHM2012 | EarlyFault | RandomForest | `compact_non_label_source` | 7 | `early_fault` | completed |

## 5. Training Completion Checks

| ID | Fit Status | Train | Val | Test | Feature Count | Metrics | Feature Importance | Status |
|---|---|---:|---:|---:|---:|---|---|---|
| Y1 | completed | 7032 | 1679 | 505 | 44 | `RMSE` | yes | done |
| Y2 | completed | 7032 | 1679 | 505 | 44 | `RMSE` | yes | done |
| Y3 | completed | 7032 | 1679 | 505 | 6 | `WeightedF1` | yes | done |
| Y4 | completed | 7032 | 1679 | 505 | 6 | `WeightedF1` | yes | done |
| Y5 | completed | 7032 | 1679 | 505 | 5 | `WeightedF1` | yes | done |
| Y6 | completed | 7032 | 1679 | 505 | 5 | `WeightedF1` | yes | done |
| Y7 | completed | 7534 | 4330 | 13025 | 7 | `RMSE` | yes | done |
| Y8 | completed | 7534 | 4330 | 13025 | 7 | `RMSE` | yes | done |
| Y9 | completed | 7534 | 4330 | 13025 | 5 | `WeightedF1` | yes | done |
| Y10 | completed | 7534 | 4330 | 13025 | 5 | `WeightedF1` | yes | done |
| Y11 | completed | 7534 | 4330 | 13025 | 7 | `WeightedF1` | yes | done |
| Y12 | completed | 7534 | 4330 | 13025 | 7 | `WeightedF1` | yes | done |

## 6. Metrics Summary

| Dataset | Task | Model | Primary | Val | Test | MLP Default | Effect |
|---|---|---|---|---:|---:|---:|---|
| XJTU-SY | rul_tabular | XGBoost | RMSE | 0.302017 | 0.431454 | 0.421645 | worse_or_equal |
| XJTU-SY | rul_tabular | RandomForest | RMSE | 0.283495 | 0.399658 | 0.421645 | improved |
| XJTU-SY | health_state_tabular | XGBoost | WeightedF1 | 0.583939 | 0.365121 | 0.371101 | worse_or_equal |
| XJTU-SY | health_state_tabular | RandomForest | WeightedF1 | 0.583314 | 0.350722 | 0.371101 | worse_or_equal |
| XJTU-SY | early_fault_tabular | XGBoost | WeightedF1 | 0.630984 | 0.839369 | 0.841682 | worse_or_equal |
| XJTU-SY | early_fault_tabular | RandomForest | WeightedF1 | 0.625932 | 0.837047 | 0.841682 | worse_or_equal |
| PHM2012 | rul_tabular | XGBoost | RMSE | 0.291317 | 0.357105 | 0.392475 | improved |
| PHM2012 | rul_tabular | RandomForest | RMSE | 0.292971 | 0.337575 | 0.392475 | improved |
| PHM2012 | health_state_tabular | XGBoost | WeightedF1 | 0.240540 | 0.295374 | 0.406725 | worse_or_equal |
| PHM2012 | health_state_tabular | RandomForest | WeightedF1 | 0.261532 | 0.340471 | 0.406725 | worse_or_equal |
| PHM2012 | early_fault_tabular | XGBoost | WeightedF1 | 0.491115 | 0.578165 | 0.664556 | worse_or_equal |
| PHM2012 | early_fault_tabular | RandomForest | WeightedF1 | 0.464725 | 0.613220 | 0.664556 | worse_or_equal |

## 7. Feature Importance

| ID | Dataset | Task | Model | Top Features |
|---|---|---|---|---|
| Y1 | XJTU-SY | RUL | XGBoost | h__time__rms (0.281), h__time__mean_abs (0.279), h__time__std (0.180), mag__time__mean (0.045), v__time__mean_abs (0.030) |
| Y2 | XJTU-SY | RUL | RandomForest | h__time__mean_abs (0.616), h__time__rms (0.153), h__time__std (0.102), h__spectral__bandwidth (0.015), mag__spectral__centroid (0.012) |
| Y3 | XJTU-SY | HealthState | XGBoost | h__time__std (0.262), h__time__rms (0.258), h__time__mean_abs (0.249), mag__time__mean (0.090), mag__time__mean_abs (0.084) |
| Y4 | XJTU-SY | HealthState | RandomForest | h__time__std (0.192), h__time__rms (0.190), mag__time__mean (0.158), mag__time__std (0.157), h__time__mean_abs (0.153) |
| Y5 | XJTU-SY | EarlyFault | XGBoost | mag__time__mean (0.397), mag__time__mean_abs (0.390), mag__time__std (0.085), v__time__std (0.080), v__time__mean_abs (0.048) |
| Y6 | XJTU-SY | EarlyFault | RandomForest | mag__time__mean_abs (0.315), mag__time__mean (0.314), v__time__mean_abs (0.149), v__time__std (0.117), mag__time__std (0.105) |
| Y7 | PHM2012 | RUL | XGBoost | h__time__rms (0.526), h__time__std (0.264), v__time__mean_abs (0.067), h__time__mean_abs (0.051), mag__time__mean_abs (0.041) |
| Y8 | PHM2012 | RUL | RandomForest | h__time__std (0.441), h__time__rms (0.195), v__time__mean_abs (0.134), h__time__mean_abs (0.104), mag__time__std (0.061) |
| Y9 | PHM2012 | HealthState | XGBoost | h__time__mean_abs (0.228), h__time__std (0.222), mag__time__mean_abs (0.189), mag__time__mean (0.184), h__time__rms (0.177) |
| Y10 | PHM2012 | HealthState | RandomForest | h__time__mean_abs (0.231), h__time__std (0.203), h__time__rms (0.190), mag__time__mean (0.189), mag__time__mean_abs (0.187) |
| Y11 | PHM2012 | EarlyFault | XGBoost | v__time__mean_abs (0.266), h__time__mean_abs (0.216), v__time__std (0.211), mag__time__mean (0.130), mag__time__mean_abs (0.060) |
| Y12 | PHM2012 | EarlyFault | RandomForest | h__time__mean_abs (0.193), v__time__mean_abs (0.182), mag__time__mean (0.153), mag__time__mean_abs (0.151), v__time__std (0.137) |

## 8. Findings

### RUL

- XJTU-SY RUL: RandomForest achieved lower test RMSE than the default independent MLP comparator, while XGBoost was slightly worse than the MLP comparator on test RMSE.
- PHM2012 RUL: both non-MLP models improved over the default independent MLP comparator; RandomForest had the best Step Y test RMSE among the two.

### HealthState

- XJTU-SY HealthState: both non-MLP classifiers were close to, but below, the default independent MLP test WeightedF1.
- PHM2012 HealthState: both non-MLP classifiers were below the default independent MLP test WeightedF1, with RandomForest stronger than XGBoost in this batch.

### EarlyFault

- XJTU-SY EarlyFault: XGBoost and RandomForest nearly matched the default independent MLP, with XGBoost slightly below the MLP comparator.
- PHM2012 EarlyFault: both non-MLP classifiers were below the default independent MLP test WeightedF1, but RandomForest outperformed XGBoost within Step Y.

## 9. Caveats

- Non-MLP models use tabular feature vectors, not raw vibration sequences.
- Step Y does not perform hyperparameter tuning, repeat seeds, reference-feature runs, full feature-grid sweeps, cross-condition runs, or tsfresh extraction.
- HealthState and EarlyFault remain pseudo-label tasks derived from degradation labeling logic.
- XGBoost and RandomForest are standalone recipe baselines and are not integrated into `ModelFactory` or `ConfigurableTrainer`.
- All Step Y independent subsets exclude the label-source reference feature policy where relevant.

## 10. Decision

- [x] Needs review: 12 real XGBoost/RandomForest tabular fits completed and curated outputs are ready for quality check.
- [ ] Needs rerun.
- [ ] Blocked.

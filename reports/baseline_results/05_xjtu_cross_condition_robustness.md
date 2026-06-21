# Step U: XJTU-SY Cross-Condition Robustness Training

## 1. Purpose

Train the recommended independent feature subsets under the XJTU-SY cross-condition split.

This step is real training, not a dry run. It runs three `mode=train` MLP experiments selected from the Step T independent recommendations and does not rerun PHM2012, tsfresh, reference subsets, or the full compact/reference grid.

## 2. Split

- train: `35Hz12kN`
- val: `37.5Hz11kN`
- test: `40Hz10kN`

This split is an operating-condition shift test. It is stricter than ordinary same-condition bearing generalization and should be read as a robustness probe.

## 3. Experiments

| ID | Task | Feature Subset | Feature Count | Run Dir | Status |
| --- | --- | --- | ---: | --- | --- |
| U1 | RUL | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-202524_xjtu_cross_rul_mlp_full_manual_basic_no_reference_0b739dcf` | completed |
| U2 | HealthState | `compact_non_label_source` | 6 | `artifacts/baselines/runs/20260621-202743_xjtu_cross_health_mlp_compact_non_label_source_ac085cd1` | completed |
| U3 | EarlyFault | `compact_non_label_source` | 5 | `artifacts/baselines/runs/20260621-203006_xjtu_cross_early_mlp_compact_non_label_source_482f884f` | completed |

## 4. Training Completion Checks

| ID | max_epochs | last_epoch | best_epoch | best_metric | checkpoints | val/test predictions | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| U1 | 50 | 50 | 45 | 0.027767 | yes | yes | pass |
| U2 | 50 | 50 | 25 | 0.721547 | yes | yes | pass |
| U3 | 50 | 50 | 18 | 0.315203 | yes | yes | pass |

Every Step U run has `history.json` length 50, final history epoch 50, `trainer_state.epoch` 50, required validation/test metrics, raw `best.ckpt` and `last.ckpt`, and raw validation/test prediction parquet files. Curated report directories copy only small review files and do not include checkpoints, predictions, manifests, features, labels, HI, or index files.

## 5. Metrics Summary

| Task | Primary Metric | Val | Test | Main-Split Test | Cross-Condition Gap | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RUL | RMSE | 0.181695 | 0.182463 | 0.421645 | -0.239183 | Cross-condition RMSE is lower than the main split reference in this run. |
| HealthState | WeightedF1 | 0.571656 | 0.702422 | 0.371101 | -0.331321 | Cross-condition WeightedF1 is higher than the main split reference in this run. |
| EarlyFault | WeightedF1 | 0.892786 | 0.754485 | 0.841682 | 0.087197 | Cross-condition WeightedF1 is worse than the main split reference. |

For RUL, RMSE is lower-is-better and the gap is `cross_test_RMSE - main_test_RMSE`, so positive means the cross-condition run is worse. For HealthState and EarlyFault, WeightedF1 is higher-is-better and the gap is `main_test_WeightedF1 - cross_test_WeightedF1`, so positive means the cross-condition run is worse.

## 6. Findings

### RUL

The cross-condition RUL run with `full_manual_basic_no_reference` reaches lower test RMSE than the main-split reference in this run. This is encouraging for the independent full-feature RUL subset, but it should not be overread as a direct same-population improvement because the cross-condition test set is much larger and comes from a different operating condition.

### HealthState

The cross-condition HealthState run with `compact_non_label_source` reaches higher test WeightedF1 than the main-split reference. The compact non-reference subset remains usable under the operating-condition shift, while still avoiding the `mag__time__rms` label-source feature.

### EarlyFault

The cross-condition EarlyFault run with `compact_non_label_source` drops below the main-split test WeightedF1. This suggests early-fault detection is more sensitive to condition shift than the other two tasks and should be treated carefully in later cross-condition reporting.

## 7. Caveats

- Cross-condition is an operating-condition shift test: train on `35Hz12kN`, validate on `37.5Hz11kN`, test on `40Hz10kN`.
- These runs exclude `mag__time__rms`; all three Step U experiments are independent non-reference runs.
- HealthState and EarlyFault are pseudo-label tasks derived from the HI/FPT labeling pipeline.
- This is still MLP only and not tuned.
- Main-split and cross-condition metrics are not perfectly apples-to-apples because their train/val/test populations and sample counts differ.

## 8. Decision

- [x] Pass to review.
- [x] Keep Step U as the XJTU-SY cross-condition robustness batch for the Step T independent recommended subsets.
- [x] Use `xjtu_cross_condition_metrics.csv` and `xjtu_cross_vs_main_comparison.csv` as the machine-readable summary tables.
- [ ] Needs rerun.
- [ ] Blocked.

Next action: Step V can merge Step T main-split summary and Step U cross-condition robustness into the final baseline report.

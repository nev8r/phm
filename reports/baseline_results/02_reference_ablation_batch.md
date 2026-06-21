# Step R: Reference Ablation Training Batch

## 1. Purpose

Train compact_with_reference MLP baselines and compare them with Step Q compact_non_label_source runs.

This step is real training, not a dry run. The only intended feature-set change is adding `mag__time__rms`, the actual HI/FPT label-source reference feature, to the compact non-label-source subset.

## 2. Experiments

| ID | Dataset | Task | Feature Count | Run Dir | Status |
| --- | --- | --- | ---: | --- | --- |
| R1 | XJTU-SY | RUL | 8 | `artifacts/baselines/runs/20260621-120732_xjtu_main_rul_mlp_compact_with_reference_04edbae4` | completed |
| R2 | XJTU-SY | HealthState | 7 | `artifacts/baselines/runs/20260621-121231_xjtu_main_health_mlp_compact_with_reference_2cef5a36` | completed |
| R3 | XJTU-SY | EarlyFault | 6 | `artifacts/baselines/runs/20260621-121715_xjtu_main_early_mlp_compact_with_reference_7225fde9` | completed |
| R4 | PHM2012 | RUL | 8 | `artifacts/baselines/runs/20260621-122203_phm_official_rul_mlp_compact_with_reference_dc5585ac` | completed |
| R5 | PHM2012 | HealthState | 6 | `artifacts/baselines/runs/20260621-122719_phm_official_health_mlp_compact_with_reference_6f46fd2f` | completed |
| R6 | PHM2012 | EarlyFault | 8 | `artifacts/baselines/runs/20260621-123242_phm_official_early_mlp_compact_with_reference_1fb5417b` | completed |

## 3. Training Completion Checks

| ID | max_epochs | last_epoch | best_epoch | best_metric | checkpoints | val/test predictions | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| R1 | 50 | 50 | 9 | 0.060795 | yes | yes | pass |
| R2 | 50 | 50 | 11 | 1.256309 | yes | yes | pass |
| R3 | 50 | 50 | 13 | 0.701994 | yes | yes | pass |
| R4 | 50 | 50 | 50 | 0.125849 | yes | yes | pass |
| R5 | 50 | 50 | 1 | 1.632760 | yes | yes | pass |
| R6 | 50 | 50 | 1 | 0.798833 | yes | yes | pass |

Every run has `history.json` length 50, final history epoch 50, `trainer_state.epoch` 50, expected feature count, and `mag__time__rms` included in `task/feature_columns.txt`.

## 4. Metrics Summary

| Experiment | Primary Metric | Val | Test | Notes |
| --- | --- | ---: | ---: | --- |
| R1 XJTU RUL | RMSE | 0.289076 | 0.468908 | lower is better |
| R2 XJTU HealthState | WeightedF1 | 0.570806 | 0.368269 | higher is better |
| R3 XJTU EarlyFault | WeightedF1 | 0.639819 | 0.841682 | higher is better |
| R4 PHM RUL | RMSE | 0.354801 | 0.360850 | lower is better |
| R5 PHM HealthState | WeightedF1 | 0.209764 | 0.417337 | higher is better |
| R6 PHM EarlyFault | WeightedF1 | 0.434481 | 0.672350 | higher is better |

## 5. Reference Ablation Comparison

| Dataset | Task | Non-reference Test | With-reference Test | Reference Effect | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| XJTU-SY | RUL RMSE | 0.428591 | 0.468908 | -0.040317 | Reference worsened test RMSE in this run. |
| XJTU-SY | HealthState WeightedF1 | 0.371101 | 0.368269 | -0.002832 | Test WeightedF1 decreased slightly. |
| XJTU-SY | EarlyFault WeightedF1 | 0.841682 | 0.841682 | 0.000000 | No meaningful test change. |
| PHM2012 | RUL RMSE | 0.392475 | 0.360850 | 0.031626 | Reference improved test RMSE. |
| PHM2012 | HealthState WeightedF1 | 0.406725 | 0.417337 | 0.010613 | Test WeightedF1 improved slightly. |
| PHM2012 | EarlyFault WeightedF1 | 0.664556 | 0.672350 | 0.007794 | Test WeightedF1 improved slightly. |

For RUL, reference effect is `non_reference_RMSE - with_reference_RMSE`, so positive means lower RMSE after adding the reference feature. For classification, reference effect is `with_reference_WeightedF1 - non_reference_WeightedF1`, so positive means higher WeightedF1 after adding the reference feature.

## 6. Findings

### RUL

The reference feature has mixed behavior. It worsens XJTU-SY test RMSE in this run but improves PHM2012 validation and test RMSE. This suggests the effect is dataset- and split-dependent, not a uniformly helpful shortcut.

### HealthState

HealthState shows small and inconsistent changes. XJTU-SY validation WeightedF1 improves while test WeightedF1 decreases slightly. PHM2012 validation WeightedF1 worsens while test WeightedF1 improves slightly.

### EarlyFault

EarlyFault is stable on XJTU-SY test and improves slightly on PHM2012. Because the added feature is part of the HI/FPT labeling path, these gains must be treated as reference-feature effects.

## 7. Caveats

- compact_with_reference includes `mag__time__rms`.
- `mag__time__rms` is the actual HI/FPT source.
- Any improvement on HealthState/EarlyFault may partly reflect pseudo-label construction.
- This is still MLP only and not a tuned model.

## 8. Decision

- [x] Pass to review.
- [ ] Needs rerun.
- [ ] Blocked.

Next action: after Step R review, proceed to Step S full feature baseline training.

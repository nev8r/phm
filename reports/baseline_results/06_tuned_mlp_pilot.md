# Step W: Tuned MLP Pilot Training

## 1. Purpose

Run a conservative tuned MLP pilot on the Step V recommended independent subsets.

This step is real training, not a dry run. It runs six `mode=train` experiments and keeps the same dataset/task coverage and independent non-reference feature subsets used by the final baseline decisions.

## 2. Tuned Setting

| Field | Default MLP | Tuned MLP |
| --- | ---: | ---: |
| hidden_size | 64 | 128 |
| batch_size | 16 | 64 |
| learning_rate | 0.001 | 0.0005 |
| weight_decay | 0.0 | 0.0001 |
| max_epochs | 50 | 50 |

## 3. Experiments

| ID | Dataset | Task | Feature Subset | Feature Count | Run Dir | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| W1 | XJTU-SY | RUL | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-215215_xjtu_main_rul_mlp_tuned_full_manual_basic_no_reference_4742cefb` | completed |
| W2 | XJTU-SY | HealthState | `compact_non_label_source` | 6 | `artifacts/baselines/runs/20260621-215630_xjtu_main_health_mlp_tuned_compact_non_label_source_500194b5` | completed |
| W3 | XJTU-SY | EarlyFault | `compact_non_label_source` | 5 | `artifacts/baselines/runs/20260621-220219_xjtu_main_early_mlp_tuned_compact_non_label_source_ff520837` | completed |
| W4 | PHM2012 | RUL | `compact_non_label_source` | 7 | `artifacts/baselines/runs/20260621-220753_phm_official_rul_mlp_tuned_compact_non_label_source_67543d68` | completed |
| W5 | PHM2012 | HealthState | `compact_non_label_source` | 5 | `artifacts/baselines/runs/20260621-221259_phm_official_health_mlp_tuned_compact_non_label_source_74d642e0` | completed |
| W6 | PHM2012 | EarlyFault | `compact_non_label_source` | 7 | `artifacts/baselines/runs/20260621-222222_phm_official_early_mlp_tuned_compact_non_label_source_1e0f18ec` | completed |

## 4. Training Completion Checks

| ID | max_epochs | last_epoch | best_epoch | best_metric | checkpoints | val/test predictions | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| W1 | 50 | 50 | 31 | 0.047420 | yes | yes | pass |
| W2 | 50 | 50 | 1 | 0.943674 | yes | yes | pass |
| W3 | 50 | 50 | 1 | 0.682924 | yes | yes | pass |
| W4 | 50 | 50 | 13 | 0.170900 | yes | yes | pass |
| W5 | 50 | 50 | 1 | 1.390988 | yes | yes | pass |
| W6 | 50 | 50 | 1 | 0.636189 | yes | yes | pass |

Every Step W run has `history.json` length 50, final history epoch 50, `trainer_state.epoch` 50, required validation/test metrics, raw `best.ckpt` and `last.ckpt`, and raw validation/test prediction parquet files. Curated report directories copy only small review files and do not include checkpoints, predictions, manifests, features, labels, HI, or index files.

## 5. Metrics Summary

| Experiment | Primary Metric | Val | Test | Default Baseline Test | Tuned Effect | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| xjtu_main_rul_mlp_tuned_full_manual_basic_no_reference | RMSE | 0.225399 | 0.443106 | 0.421645 | -0.021461 | Tuned MLP worsened the test primary metric relative to the default baseline. |
| xjtu_main_health_mlp_tuned_compact_non_label_source | WeightedF1 | 0.584941 | 0.359146 | 0.371101 | -0.011955 | Tuned MLP worsened the test primary metric relative to the default baseline. |
| xjtu_main_early_mlp_tuned_compact_non_label_source | WeightedF1 | 0.644101 | 0.837047 | 0.841682 | -0.004635 | Tuned MLP worsened the test primary metric relative to the default baseline. |
| phm_official_rul_mlp_tuned_compact_non_label_source | RMSE | 0.468337 | 0.337661 | 0.392475 | 0.054814 | Tuned MLP improved the test primary metric over the default baseline. |
| phm_official_health_mlp_tuned_compact_non_label_source | WeightedF1 | 0.238534 | 0.441598 | 0.406725 | 0.034874 | Tuned MLP improved the test primary metric over the default baseline. |
| phm_official_early_mlp_tuned_compact_non_label_source | WeightedF1 | 0.372244 | 0.679212 | 0.664556 | 0.014657 | Tuned MLP improved the test primary metric over the default baseline. |

For RUL, RMSE is lower-is-better and `tuned_effect = default_RMSE - tuned_RMSE`, so positive means tuned is better. For HealthState and EarlyFault, WeightedF1 is higher-is-better and `tuned_effect = tuned_WeightedF1 - default_WeightedF1`, so positive means tuned is better.

## 6. Findings

### RUL

- XJTU-SY RUL tuned effect = -0.021461 on RMSE. Tuned MLP worsened the test primary metric relative to the default baseline.
- PHM2012 RUL tuned effect = 0.054814 on RMSE. Tuned MLP improved the test primary metric over the default baseline.

### HealthState

- XJTU-SY HealthState tuned effect = -0.011955 on WeightedF1. Tuned MLP worsened the test primary metric relative to the default baseline.
- PHM2012 HealthState tuned effect = 0.034874 on WeightedF1. Tuned MLP improved the test primary metric over the default baseline.

### EarlyFault

- XJTU-SY EarlyFault tuned effect = -0.004635 on WeightedF1. Tuned MLP worsened the test primary metric relative to the default baseline.
- PHM2012 EarlyFault tuned effect = 0.014657 on WeightedF1. Tuned MLP improved the test primary metric over the default baseline.

## 7. Caveats

- Tuned MLP is a pilot, not a full hyperparameter search.
- All runs use independent non-reference subsets.
- HealthState and EarlyFault are pseudo-label tasks.
- No cross-condition tuned run is included yet.
- Any replacement of default MLP with tuned MLP should consider validation/test consistency, not only a single test metric.

## 8. Decision

- [x] Pass to review.
- [x] Keep Step W as a conservative tuned MLP pilot on Step V recommended independent subsets.
- [x] Use `tuned_mlp_pilot_metrics.csv` and `tuned_vs_default_mlp_comparison.csv` as the machine-readable summary tables.
- [ ] Needs rerun.
- [ ] Blocked.

Next action: review whether tuned MLP should replace default MLP for any dataset/task, or whether the next phase should move to non-MLP tabular baselines.

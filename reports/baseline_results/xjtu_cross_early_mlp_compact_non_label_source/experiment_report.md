# xjtu_cross_early_mlp_compact_non_label_source

## Status

- Status: completed, ready for Step U review.
- Raw run directory: `artifacts/baselines/runs/20260621-203006_xjtu_cross_early_mlp_compact_non_label_source_482f884f`
- Curated report directory: `reports/baseline_results/xjtu_cross_early_mlp_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_cross_condition`.
- Task: `early_fault_tabular`.
- Task type: binary_classification.
- Target: `early_fault`.
- Feature subset: `compact_non_label_source`.
- Label source included: no.
- Feature count: 5.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 9216.
- Train examples: 616.
- Validation examples: 1566.
- Test examples: 7034.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 18.
- Best metric: 0.31520273431831464.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.893359 | 0.892653 | 0.892786 | 0.370237 |
| Test | 0.744669 | 0.721620 | 0.754485 | 0.724707 |

## Cross-Condition Caveat

This run trains on 35Hz12kN, validates on 37.5Hz11kN, and tests on 40Hz10kN. Results measure condition-shift robustness, not ordinary same-condition bearing generalization.

## Decision

- Status: keep for Step U cross-condition robustness review.
- Primary comparison metric: WeightedF1, higher is better.
- Decision: Keep as the XJTU-SY cross-condition independent EarlyFault robustness run; WeightedF1 is higher-is-better.

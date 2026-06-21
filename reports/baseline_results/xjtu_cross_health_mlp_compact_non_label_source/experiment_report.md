# xjtu_cross_health_mlp_compact_non_label_source

## Status

- Status: completed, ready for Step U review.
- Raw run directory: `artifacts/baselines/runs/20260621-202743_xjtu_cross_health_mlp_compact_non_label_source_ac085cd1`
- Curated report directory: `reports/baseline_results/xjtu_cross_health_mlp_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_cross_condition`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
- Feature subset: `compact_non_label_source`.
- Label source included: no.
- Feature count: 6.
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
- Best epoch: 25.
- Best metric: 0.7215468736783582.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.604087 | 0.505109 | 0.571656 | 0.744211 |
| Test | 0.680694 | 0.514514 | 0.702422 | 0.846511 |

## Cross-Condition Caveat

This run trains on 35Hz12kN, validates on 37.5Hz11kN, and tests on 40Hz10kN. Results measure condition-shift robustness, not ordinary same-condition bearing generalization.

## Decision

- Status: keep for Step U cross-condition robustness review.
- Primary comparison metric: WeightedF1, higher is better.
- Decision: Keep as the XJTU-SY cross-condition independent HealthState robustness run; WeightedF1 is higher-is-better.

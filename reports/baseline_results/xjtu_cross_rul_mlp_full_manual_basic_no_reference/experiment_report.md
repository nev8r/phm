# xjtu_cross_rul_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for Step U review.
- Raw run directory: `artifacts/baselines/runs/20260621-202524_xjtu_cross_rul_mlp_full_manual_basic_no_reference_0b739dcf`
- Curated report directory: `reports/baseline_results/xjtu_cross_rul_mlp_full_manual_basic_no_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_cross_condition`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
- Feature subset: `full_manual_basic_no_reference`.
- Label source included: no.
- Feature count: 44.
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
- Best epoch: 45.
- Best metric: 0.027766565175736512.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.145692 | 0.181695 | 0.033033 |
| Test | 0.127480 | 0.182463 | 0.033535 |

## Cross-Condition Caveat

This run trains on 35Hz12kN, validates on 37.5Hz11kN, and tests on 40Hz10kN. Results measure condition-shift robustness, not ordinary same-condition bearing generalization.

## Decision

- Status: keep for Step U cross-condition robustness review.
- Primary comparison metric: RMSE, lower is better.
- Decision: Keep as the XJTU-SY cross-condition independent RUL robustness run; RMSE is lower-is-better.

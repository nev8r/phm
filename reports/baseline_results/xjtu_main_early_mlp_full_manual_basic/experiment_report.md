# xjtu_main_early_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-171716_xjtu_main_early_mlp_full_manual_basic_8c2c8626`
- Curated report directory: `reports/baseline_results/xjtu_main_early_mlp_full_manual_basic`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `early_fault_tabular`.
- Task type: binary_classification.
- Target: `early_fault`.
- Feature subset: `full_manual_basic`.
- Label source included: yes.
- Feature count: 45.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 9216.
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 1.
- Best metric: 1.0007533056029383.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.733175 | 0.603976 | 0.675783 | 1.913776 |
| Test | 0.663366 | 0.515475 | 0.577494 | 10.239550 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

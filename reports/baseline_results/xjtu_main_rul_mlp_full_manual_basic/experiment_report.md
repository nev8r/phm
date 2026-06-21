# xjtu_main_rul_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-165829_xjtu_main_rul_mlp_full_manual_basic_d0c53668`
- Curated report directory: `reports/baseline_results/xjtu_main_rul_mlp_full_manual_basic`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
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
- Best epoch: 7.
- Best metric: 0.0490004868634666.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.171694 | 0.462780 | 0.214056 |
| Test | 0.394571 | 0.440693 | 0.191899 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: RMSE, lower is better.

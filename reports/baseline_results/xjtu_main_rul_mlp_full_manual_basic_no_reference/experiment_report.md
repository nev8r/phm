# xjtu_main_rul_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-165326_xjtu_main_rul_mlp_full_manual_basic_no_reference_c0d6f3ac`
- Curated report directory: `reports/baseline_results/xjtu_main_rul_mlp_full_manual_basic_no_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
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
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 13.
- Best metric: 0.046903513447891565.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.161645 | 0.339505 | 0.115219 |
| Test | 0.376273 | 0.421645 | 0.175789 |

## Caveat

This run excludes mag__time__rms; it can be treated as an independent non-reference full-feature run.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: RMSE, lower is better.

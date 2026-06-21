# xjtu_main_rul_mlp_tuned_full_manual_basic_no_reference

## Status

- Status: completed, ready for Step W review.
- Raw run directory: `artifacts/baselines/runs/20260621-215215_xjtu_main_rul_mlp_tuned_full_manual_basic_no_reference_4742cefb`
- Curated report directory: `reports/baseline_results/xjtu_main_rul_mlp_tuned_full_manual_basic_no_reference`

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

## Tuned Setting

- `model.params.hidden_size`: 128.
- `trainer.batch_size`: 64.
- `trainer.optimizer.lr`: 0.0005.
- `trainer.optimizer.weight_decay`: 0.0001.

## Data Shape

- Examples: 9216.
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 31.
- Best metric: 0.04742049031106203.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.150404 | 0.225399 | 0.051500 |
| Test | 0.385476 | 0.443106 | 0.198018 |

## Caveat

This is a conservative tuned MLP pilot, not a full hyperparameter search. It uses the Step V independent non-reference feature subset and does not include `mag__time__rms`.

## Decision

- Status: keep for Step W tuned MLP pilot review.
- Primary comparison metric: RMSE, lower is better.

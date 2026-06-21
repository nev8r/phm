# xjtu_main_early_mlp_tuned_compact_non_label_source

## Status

- Status: completed, ready for Step W review.
- Raw run directory: `artifacts/baselines/runs/20260621-220219_xjtu_main_early_mlp_tuned_compact_non_label_source_ff520837`
- Curated report directory: `reports/baseline_results/xjtu_main_early_mlp_tuned_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `early_fault_tabular`.
- Task type: binary_classification.
- Target: `early_fault`.
- Feature subset: `compact_non_label_source`.
- Label source included: no.
- Feature count: 5.
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
- Best epoch: 1.
- Best metric: 0.6829241250583714.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.718285 | 0.560506 | 0.644101 | 0.791348 |
| Test | 0.847525 | 0.821127 | 0.837047 | 3.691874 |

## Caveat

This is a conservative tuned MLP pilot, not a full hyperparameter search. It uses the Step V independent non-reference feature subset and does not include `mag__time__rms`.

## Decision

- Status: keep for Step W tuned MLP pilot review.
- Primary comparison metric: WeightedF1, higher is better.

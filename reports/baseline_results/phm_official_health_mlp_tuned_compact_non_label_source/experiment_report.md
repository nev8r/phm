# phm_official_health_mlp_tuned_compact_non_label_source

## Status

- Status: completed, ready for Step W review.
- Raw run directory: `artifacts/baselines/runs/20260621-221259_phm_official_health_mlp_tuned_compact_non_label_source_74d642e0`
- Curated report directory: `reports/baseline_results/phm_official_health_mlp_tuned_compact_non_label_source`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
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

- Examples: 24889.
- Train examples: 7534.
- Validation examples: 4330.
- Test examples: 13025.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 1.
- Best metric: 1.3909879577729631.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.251039 | 0.250477 | 0.238534 | 2.100134 |
| Test | 0.480845 | 0.324147 | 0.441598 | 1.676448 |

## Caveat

This is a conservative tuned MLP pilot, not a full hyperparameter search. It uses the Step V independent non-reference feature subset and does not include `mag__time__rms`.

## Decision

- Status: keep for Step W tuned MLP pilot review.
- Primary comparison metric: WeightedF1, higher is better.

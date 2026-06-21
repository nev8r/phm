# xjtu_main_health_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-170311_xjtu_main_health_mlp_full_manual_basic_no_reference_9ebfd46d`
- Curated report directory: `reports/baseline_results/xjtu_main_health_mlp_full_manual_basic_no_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
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
- Best epoch: 1.
- Best metric: 1.3801005005969533.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.701608 | 0.414395 | 0.610433 | 4.975988 |
| Test | 0.257426 | 0.248964 | 0.227696 | 8.738139 |

## Caveat

This run excludes mag__time__rms; it can be treated as an independent non-reference full-feature run.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

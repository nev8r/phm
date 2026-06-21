# xjtu_main_health_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-170756_xjtu_main_health_mlp_full_manual_basic_2ed9942f`
- Curated report directory: `reports/baseline_results/xjtu_main_health_mlp_full_manual_basic`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
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
- Best metric: 1.3393372853308738.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.685527 | 0.354899 | 0.597020 | 4.750750 |
| Test | 0.253465 | 0.233359 | 0.213177 | 10.549318 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

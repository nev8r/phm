# phm_official_health_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-173551_phm_official_health_mlp_full_manual_basic_1e660f03`
- Curated report directory: `reports/baseline_results/phm_official_health_mlp_full_manual_basic`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
- Feature subset: `full_manual_basic`.
- Label source included: yes.
- Feature count: 45.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

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
- Best metric: 1.6190389373556386.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.311547 | 0.249541 | 0.270328 | 5.809330 |
| Test | 0.310403 | 0.311597 | 0.304306 | 6.235229 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

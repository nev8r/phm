# phm_official_rul_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-172607_phm_official_rul_mlp_full_manual_basic_d8241feb`
- Curated report directory: `reports/baseline_results/phm_official_rul_mlp_full_manual_basic`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
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
- Best metric: 0.11986977232313402.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.335621 | 0.421059 | 0.178990 |
| Test | 0.269185 | 0.334945 | 0.112065 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: RMSE, lower is better.

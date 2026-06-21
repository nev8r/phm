# phm_official_rul_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-172133_phm_official_rul_mlp_full_manual_basic_no_reference_3ad07f96`
- Curated report directory: `reports/baseline_results/phm_official_rul_mlp_full_manual_basic_no_reference`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
- Feature subset: `full_manual_basic_no_reference`.
- Label source included: no.
- Feature count: 44.
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
- Best epoch: 31.
- Best metric: 0.1521411155482816.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.337142 | 0.474238 | 0.229471 |
| Test | 0.272236 | 0.408619 | 0.166782 |

## Caveat

This run excludes mag__time__rms; it can be treated as an independent non-reference full-feature run.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: RMSE, lower is better.

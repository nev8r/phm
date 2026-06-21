# phm_official_early_mlp_full_manual_basic

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-174612_phm_official_early_mlp_full_manual_basic_aead21f0`
- Curated report directory: `reports/baseline_results/phm_official_early_mlp_full_manual_basic`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `early_fault_tabular`.
- Task type: binary_classification.
- Target: `early_fault`.
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
- Best epoch: 2.
- Best metric: 0.5009747552516024.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.524018 | 0.523860 | 0.526063 | 4.415832 |
| Test | 0.549021 | 0.485614 | 0.487209 | 5.578041 |

## Caveat

This run includes mag__time__rms; any gain on HealthState/EarlyFault may reflect HI/FPT label-source shortcut.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

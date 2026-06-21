# phm_official_early_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-174100_phm_official_early_mlp_full_manual_basic_no_reference_d53b5bba`
- Curated report directory: `reports/baseline_results/phm_official_early_mlp_full_manual_basic_no_reference`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `early_fault_tabular`.
- Task type: binary_classification.
- Target: `early_fault`.
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
- Best epoch: 1.
- Best metric: 0.5133701571347361.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.524249 | 0.524070 | 0.526412 | 4.602102 |
| Test | 0.538042 | 0.470830 | 0.472495 | 5.809269 |

## Caveat

This run excludes mag__time__rms; it can be treated as an independent non-reference full-feature run.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

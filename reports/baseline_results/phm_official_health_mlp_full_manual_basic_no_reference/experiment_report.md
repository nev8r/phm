# phm_official_health_mlp_full_manual_basic_no_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-173102_phm_official_health_mlp_full_manual_basic_no_reference_490e9ded`
- Curated report directory: `reports/baseline_results/phm_official_health_mlp_full_manual_basic_no_reference`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `health_state_tabular`.
- Task type: multiclass_classification.
- Target: `health_state_id`.
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
- Best metric: 1.7968476433842266.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.317321 | 0.273749 | 0.290378 | 6.874701 |
| Test | 0.333973 | 0.331311 | 0.339013 | 6.167757 |

## Caveat

This run excludes mag__time__rms; it can be treated as an independent non-reference full-feature run.

## Decision

- Status: keep for Step S full feature baseline review.
- Primary comparison metric: WeightedF1, higher is better.

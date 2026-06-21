# phm_official_health_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-101716_phm_official_health_mlp_compact_non_label_source_2e7f27ff`
- Curated report directory: `reports/baseline_results/phm_official_health_mlp_compact_non_label_source`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `health_state_tabular`.
- Task type: multiclass classification.
- Target: `health_state_id`.
- Feature set: `manual_basic`, compact non-label-source subset.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 24889.
- Train examples: 7534.
- Validation examples: 4330.
- Test examples: 13025.
- Feature columns: 5.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 1.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.275058 | 0.268575 | 0.258953 | 2.329762 |
| Test | 0.432860 | 0.295758 | 0.406725 | 1.789757 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

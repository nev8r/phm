# phm_official_early_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-102318_phm_official_early_mlp_compact_non_label_source_6ddffda7`
- Curated report directory: `reports/baseline_results/phm_official_early_mlp_compact_non_label_source`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `early_fault_tabular`.
- Task type: binary classification.
- Target: `early_fault`.
- Feature set: `manual_basic`, compact non-label-source subset.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 24889.
- Train examples: 7534.
- Validation examples: 4330.
- Test examples: 13025.
- Feature columns: 7.

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
| Validation | 0.420554 | 0.409017 | 0.429956 | 1.367257 |
| Test | 0.672553 | 0.664085 | 0.664556 | 0.810184 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

# phm_official_rul_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-101138_phm_official_rul_mlp_compact_non_label_source_42d12fa7`
- Curated report directory: `reports/baseline_results/phm_official_rul_mlp_compact_non_label_source`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
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
- Best epoch: 50.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.261271 | 0.369824 | 0.137695 |
| Test | 0.277914 | 0.392475 | 0.154015 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

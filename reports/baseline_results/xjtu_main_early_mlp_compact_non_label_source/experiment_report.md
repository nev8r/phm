# xjtu_main_early_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-100645_xjtu_main_early_mlp_compact_non_label_source_4750af38`
- Curated report directory: `reports/baseline_results/xjtu_main_early_mlp_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `early_fault_tabular`.
- Task type: binary classification.
- Target: `early_fault`.
- Feature set: `manual_basic`, compact non-label-source subset.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 9216.
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.
- Feature columns: 5.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 13.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.717094 | 0.557662 | 0.641965 | 0.979196 |
| Test | 0.851485 | 0.826390 | 0.841682 | 5.395761 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

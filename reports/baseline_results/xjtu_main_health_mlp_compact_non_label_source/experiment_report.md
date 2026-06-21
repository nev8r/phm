# xjtu_main_health_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-100202_xjtu_main_health_mlp_compact_non_label_source_f9db9192`
- Curated report directory: `reports/baseline_results/xjtu_main_health_mlp_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `health_state_tabular`.
- Task type: multiclass classification.
- Target: `health_state_id`.
- Feature set: `manual_basic`, compact non-label-source subset.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 9216.
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.
- Feature columns: 6.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 11.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.666468 | 0.251763 | 0.554834 | 2.116033 |
| Test | 0.362376 | 0.308675 | 0.371101 | 6.571516 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

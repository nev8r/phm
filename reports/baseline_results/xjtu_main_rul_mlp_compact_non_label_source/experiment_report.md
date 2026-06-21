# xjtu_main_rul_mlp_compact_non_label_source

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-095716_xjtu_main_rul_mlp_compact_non_label_source_1fce0054`
- Curated report directory: `reports/baseline_results/xjtu_main_rul_mlp_compact_non_label_source`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
- Feature set: `manual_basic`, compact non-label-source subset.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Data Shape

- Examples: 9216.
- Train examples: 7032.
- Validation examples: 1679.
- Test examples: 505.
- Feature columns: 7.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 9.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.153175 | 0.284873 | 0.081284 |
| Test | 0.339655 | 0.428591 | 0.190489 |

## Curation Rule

Only small review files are copied here. Checkpoints, prediction parquet files, task manifests, raw features, cleaned features, labels, HI files, and index files remain under the raw artifact directory.

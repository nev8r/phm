# xjtu_main_rul_mlp_compact_with_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-120732_xjtu_main_rul_mlp_compact_with_reference_04edbae4`
- Curated report directory: `reports/baseline_results/xjtu_main_rul_mlp_compact_with_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `rul_tabular`.
- Task type: regression.
- Target: `piecewise_rul_norm`.
- Feature subset: `compact_with_reference`.
- Label source included: yes, `mag__time__rms`.
- Feature count: 8.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 9.
- Best metric: 0.060795125109559325.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | MAE | RMSE | Loss |
| --- | ---: | ---: | ---: |
| Validation | 0.149620 | 0.289076 | 0.083691 |
| Test | 0.368809 | 0.468908 | 0.223476 |

## Caveat

This run includes mag__time__rms, the actual HI/FPT label-source feature. Performance gain over compact_non_label_source must be interpreted as reference-feature effect, not independent feature evidence.

## Decision

- Status: keep for Step R reference ablation review.
- Primary comparison metric: RMSE, lower is better.

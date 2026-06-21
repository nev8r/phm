# xjtu_main_early_mlp_compact_with_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-121715_xjtu_main_early_mlp_compact_with_reference_7225fde9`
- Curated report directory: `reports/baseline_results/xjtu_main_early_mlp_compact_with_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `early_fault_tabular`.
- Task type: binary classification.
- Target: `early_fault`.
- Feature subset: `compact_with_reference`.
- Label source included: yes, `mag__time__rms`.
- Feature count: 6.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 13.
- Best metric: 0.7019937384339983.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.715902 | 0.554803 | 0.639819 | 0.975982 |
| Test | 0.851485 | 0.826390 | 0.841682 | 4.997978 |

## Caveat

This run includes mag__time__rms, the actual HI/FPT label-source feature. Performance gain over compact_non_label_source must be interpreted as reference-feature effect, not independent feature evidence.

## Decision

- Status: keep for Step R reference ablation review.
- Primary comparison metric: WeightedF1, higher is better.

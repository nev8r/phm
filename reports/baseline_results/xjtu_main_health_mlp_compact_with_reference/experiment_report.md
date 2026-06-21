# xjtu_main_health_mlp_compact_with_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-121231_xjtu_main_health_mlp_compact_with_reference_2cef5a36`
- Curated report directory: `reports/baseline_results/xjtu_main_health_mlp_compact_with_reference`

## Task

- Dataset: XJTU-SY.
- Split: `xjtu_bearing_index_split`.
- Task: `health_state_tabular`.
- Task type: multiclass classification.
- Target: `health_state_id`.
- Feature subset: `compact_with_reference`.
- Label source included: yes, `mag__time__rms`.
- Feature count: 7.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 11.
- Best metric: 1.2563086428457781.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.677189 | 0.307490 | 0.570806 | 1.967252 |
| Test | 0.358416 | 0.304254 | 0.368269 | 6.795975 |

## Caveat

This run includes mag__time__rms, the actual HI/FPT label-source feature. Performance gain over compact_non_label_source must be interpreted as reference-feature effect, not independent feature evidence.

## Decision

- Status: keep for Step R reference ablation review.
- Primary comparison metric: WeightedF1, higher is better.

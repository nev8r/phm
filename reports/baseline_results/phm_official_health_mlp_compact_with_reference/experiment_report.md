# phm_official_health_mlp_compact_with_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-122719_phm_official_health_mlp_compact_with_reference_6f46fd2f`
- Curated report directory: `reports/baseline_results/phm_official_health_mlp_compact_with_reference`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `health_state_tabular`.
- Task type: multiclass classification.
- Target: `health_state_id`.
- Feature subset: `compact_with_reference`.
- Label source included: yes, `mag__time__rms`.
- Feature count: 6.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 1.
- Best metric: 1.6327598329244248.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.239030 | 0.212327 | 0.209764 | 2.615505 |
| Test | 0.421574 | 0.321112 | 0.417337 | 1.761478 |

## Caveat

This run includes mag__time__rms, the actual HI/FPT label-source feature. Performance gain over compact_non_label_source must be interpreted as reference-feature effect, not independent feature evidence.

## Decision

- Status: keep for Step R reference ablation review.
- Primary comparison metric: WeightedF1, higher is better.

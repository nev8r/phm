# phm_official_early_mlp_compact_with_reference

## Status

- Status: completed, ready for review.
- Raw run directory: `artifacts/baselines/runs/20260621-123242_phm_official_early_mlp_compact_with_reference_1fb5417b`
- Curated report directory: `reports/baseline_results/phm_official_early_mlp_compact_with_reference`

## Task

- Dataset: PHM2012.
- Split: `phm2012_official`.
- Task: `early_fault_tabular`.
- Task type: binary classification.
- Target: `early_fault`.
- Feature subset: `compact_with_reference`.
- Label source included: yes, `mag__time__rms`.
- Feature count: 8.
- Model: `mlp`.
- Trainer: `base`, 50 epochs.

## Training Check

- `history.json` entries: 50.
- Last epoch: 50.
- `trainer_state.epoch`: 50.
- Best epoch: 1.
- Best metric: 0.7988331925666009.
- Checkpoints exist in raw artifact: `best.ckpt`, `last.ckpt`.
- Prediction parquet files exist in raw artifact: validation and test.

## Metrics

| Split | Accuracy | MacroF1 | WeightedF1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.425173 | 0.413495 | 0.434481 | 1.351324 |
| Test | 0.681305 | 0.671858 | 0.672350 | 0.761941 |

## Caveat

This run includes mag__time__rms, the actual HI/FPT label-source feature. Performance gain over compact_non_label_source must be interpreted as reference-feature effect, not independent feature evidence.

## Decision

- Status: keep for Step R reference ablation review.
- Primary comparison metric: WeightedF1, higher is better.

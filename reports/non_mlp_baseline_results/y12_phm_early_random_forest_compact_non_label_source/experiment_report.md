# y12_phm_early_random_forest_compact_non_label_source

This is a real standalone non-MLP tabular baseline fit.

## Run

- Dataset: `PHM2012`
- Split: `phm2012_official`
- Task: `early_fault_tabular`
- Model: `random_forest_classifier`
- Fit status: `completed`
- Feature count: 7
- Target columns: `early_fault`
- Train / Val / Test examples: 7534 / 4330 / 13025

## Metrics

- Primary metric: `WeightedF1` (higher_is_better)
- Val primary: 0.464725
- Test primary: 0.613220

| Split | Metric | Value |
|---|---|---:|
| val | Accuracy | 0.455889 |
| val | MacroF1 | 0.448112 |
| val | WeightedF1 | 0.464725 |
| test | Accuracy | 0.624491 |
| test | MacroF1 | 0.612621 |
| test | WeightedF1 | 0.613220 |

## Top Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `h__time__mean_abs` | 0.193068 |
| 2 | `v__time__mean_abs` | 0.181908 |
| 3 | `mag__time__mean` | 0.153465 |
| 4 | `mag__time__mean_abs` | 0.151087 |
| 5 | `v__time__std` | 0.136876 |
| 6 | `h__time__rms` | 0.095209 |
| 7 | `h__time__std` | 0.088388 |

## Caveats

- This model uses tabular manual features, not raw vibration sequences.
- No hyperparameter sweep is performed in Step Y.
- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.

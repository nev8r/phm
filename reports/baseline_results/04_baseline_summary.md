# Step T: Baseline Summary and Main-Split Decisions

## 1. Purpose

Summarize the real training results from Step Q, Step R, and Step S. This step does not run training, evaluation, feature extraction, prediction export, or checkpoint creation. It only consolidates the curated result files already stored under `reports/baseline_results/`.

The goal is to close the main-split / official-split baseline cycle by separating three different ideas:

- best metric subset: the feature subset with the strongest test metric in the current grid.
- independent recommended subset: the subset recommended for ordinary baseline reporting when label-source features should be avoided.
- reference/sanity subset: a subset that includes `mag__time__rms`, the HI/FPT label-source feature, and therefore should not be treated as independent evidence for HealthState or EarlyFault.

## 2. Input Files

| File | Role |
| --- | --- |
| `first_training_batch_metrics.csv` | Step Q compact non-reference MLP results. |
| `reference_ablation_metrics.csv` | Step R compact-with-reference MLP results. |
| `reference_ablation_comparison.csv` | Step R reference-effect comparison. |
| `full_feature_batch_metrics.csv` | Step S full manual-basic MLP results. |
| `full_vs_compact_comparison.csv` | Step S compact-vs-full comparison. |
| `01_first_training_batch.md` | Step Q narrative report. |
| `02_reference_ablation_batch.md` | Step R narrative report. |
| `03_full_feature_batch.md` | Step S narrative report. |

## 3. Completed Training Matrix

The consolidated matrix contains 24 completed MLP runs:

| Dataset | Tasks | Feature Subsets | Runs |
| --- | --- | --- | ---: |
| XJTU-SY | RUL, HealthState, EarlyFault | compact non-reference, compact reference, full non-reference, full reference | 12 |
| PHM2012 | RUL, HealthState, EarlyFault | compact non-reference, compact reference, full non-reference, full reference | 12 |
| Total | 6 dataset-task pairs | 4 subsets each | 24 |

All rows are written to `baseline_all_results.csv` with one row per trained experiment.

## 4. Best Test Results by Dataset/Task

| Dataset | Task | Best Metric Subset | Reference? | Test Metric | Runner-up | Decision |
| --- | --- | --- | --- | ---: | --- | --- |
| XJTU-SY | RUL | `full_manual_basic_no_reference` | no | 0.421645 RMSE | `compact_non_label_source` | use best non-reference subset as the main independent baseline. |
| XJTU-SY | HealthState | `compact_non_label_source` | no | 0.371101 WeightedF1 | `compact_with_reference` | use best non-reference subset as the main independent baseline. |
| XJTU-SY | EarlyFault | `compact_non_label_source` | no | 0.841682 WeightedF1 | `compact_with_reference` | use best non-reference subset as the main independent baseline. |
| PHM2012 | RUL | `full_manual_basic` | yes | 0.334945 RMSE | `compact_with_reference` | best metric includes reference feature; independent conclusion should compare non-reference candidate. |
| PHM2012 | HealthState | `compact_with_reference` | yes | 0.417337 WeightedF1 | `compact_non_label_source` | best metric includes reference feature; independent conclusion should compare non-reference candidate. |
| PHM2012 | EarlyFault | `compact_with_reference` | yes | 0.672350 WeightedF1 | `compact_non_label_source` | best metric includes reference feature; independent conclusion should compare non-reference candidate. |

For RUL, lower RMSE is better. For HealthState and EarlyFault, higher WeightedF1 is better. When a reference subset wins the best metric, the result is retained as the best metric result but not promoted to the independent baseline conclusion.

## 5. Compact vs Full Findings

| Dataset | Task | Compact Non-ref | Compact Ref | Full Non-ref | Full Ref | Main Reading |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| XJTU-SY | RUL | 0.428591 | 0.468908 | 0.421645 | 0.440693 | Full non-reference is slightly better than compact non-reference; reference features are unstable and are not the default independent choice. |
| XJTU-SY | HealthState | 0.371101 | 0.368269 | 0.227696 | 0.213177 | Compact non-reference is the best independent and best overall metric; full feature sets degrade test WeightedF1. |
| XJTU-SY | EarlyFault | 0.841682 | 0.841682 | 0.576118 | 0.577494 | Compact non-reference ties compact reference on test WeightedF1 and clearly beats full feature sets; choose compact non-reference as independent baseline. |
| PHM2012 | RUL | 0.392475 | 0.360850 | 0.408619 | 0.334945 | Reference features improve the best metric, especially in the full set, but the independent non-reference comparison favors compact over full. |
| PHM2012 | HealthState | 0.406725 | 0.417337 | 0.339013 | 0.304306 | Compact subsets beat full subsets; compact reference is best metric, while compact non-reference is the independent recommendation. |
| PHM2012 | EarlyFault | 0.664556 | 0.672350 | 0.472495 | 0.487209 | Compact subsets beat full subsets; compact reference is best metric, while compact non-reference remains the independent recommendation. |

Compact subsets are the safer default for HealthState and EarlyFault on both datasets. Full features hurt the classification tasks in this MLP setting, which likely reflects redundant or noisy features relative to the small compact task-specific sets.

For RUL, the result is dataset dependent. XJTU-SY benefits slightly from the full non-reference set, while PHM2012's independent non-reference comparison favors the compact set. PHM2012's strongest RUL metric appears only when the reference feature is included, so it should be reported as a reference/sanity result.

## 6. Reference Feature Findings

`mag__time__rms` is a useful diagnostic feature, but it is also the feature used by the HI/FPT labeling pipeline. This creates a label-source caveat for HealthState and EarlyFault and a possible shortcut for any supervised task that uses those derived labels.

The reference effect is not uniformly beneficial:

- XJTU-SY RUL worsens when the compact reference feature is added and remains worse than the full non-reference run.
- PHM2012 RUL improves with reference features, especially in the full reference subset.
- HealthState reference effects are small or inconsistent, and compact subsets remain stronger than full subsets.
- EarlyFault compact reference ties XJTU-SY and slightly improves PHM2012, but the independent compact subset remains the reporting default.

## 7. Recommended Baseline Feature Subsets

| Dataset | Task | Recommended Independent Subset | Best Metric Subset | Reference/Sanity Note |
| --- | --- | --- | --- | --- |
| XJTU-SY | RUL | `full_manual_basic_no_reference` | `full_manual_basic_no_reference` (0.421645) | No reference subset is recommended as the main baseline for this task. |
| XJTU-SY | HealthState | `compact_non_label_source` | `compact_non_label_source` (0.371101) | Reference effect is small and inconsistent. |
| XJTU-SY | EarlyFault | `compact_non_label_source` | `compact_non_label_source` (0.841682) | The reference subset ties the best metric but adds no independent gain. |
| PHM2012 | RUL | `compact_non_label_source` | `full_manual_basic` (0.334945) | full_manual_basic is the best metric subset but is a reference/sanity result because it includes mag__time__rms. |
| PHM2012 | HealthState | `compact_non_label_source` | `compact_with_reference` (0.417337) | compact_with_reference is a reference/sanity result, not independent evidence. |
| PHM2012 | EarlyFault | `compact_non_label_source` | `compact_with_reference` (0.672350) | compact_with_reference is a reference/sanity result, not independent evidence. |

These recommendations are for the current MLP main-split / official-split baseline only. They should not be generalized to cross-condition XJTU-SY evaluation or non-MLP model families without another run.

## 8. Key Caveats

- HealthState and EarlyFault are pseudo-label tasks derived from the HI/FPT labeling pipeline.
- `compact_with_reference` and `full_manual_basic` include `mag__time__rms`; use them as reference/sanity results, not as independent evidence.
- The current cycle uses MLP baselines only and does not tune architectures or hyperparameters.
- These are main split / official split conclusions: XJTU-SY cross-condition robustness remains a separate question.
- This summary is based on curated report CSVs under `reports/baseline_results/`; raw checkpoints, predictions, feature tables, labels, and manifests remain outside the committed report tree.

## 9. Decision

- [x] Pass to review.
- [x] Close the main-split / official-split MLP baseline summary using Step Q/R/S results.
- [x] Use `baseline_all_results.csv`, `baseline_best_by_task.csv`, and `baseline_feature_subset_comparison.csv` as the machine-readable summary tables.
- [ ] Needs rerun.
- [ ] Blocked.

Next action: either close the current main-split baseline cycle or open optional Step U for XJTU-SY cross-condition robustness with the recommended independent subsets.

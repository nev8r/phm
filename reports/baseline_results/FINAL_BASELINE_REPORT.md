# Final Baseline Report

## 1. Scope

This report summarizes the completed MLP baseline cycle for the current PHM project branch.

It includes:

- Step Q: compact non-reference MLP baselines.
- Step R: compact with-reference ablation baselines.
- Step S: full manual feature set baselines.
- Step T: main split / official split summary.
- Step U: XJTU-SY cross-condition robustness.

No new training, evaluation, checkpoint creation, prediction export, feature extraction, or raw artifact copy is run in this step. Step V only consolidates curated results already stored under `reports/baseline_results/`.

## 2. Completed Training Runs

| Stage | Runs | Purpose |
| --- | ---: | --- |
| Step Q | 6 | Compact non-reference subsets for XJTU-SY and PHM2012 across RUL, HealthState, and EarlyFault. |
| Step R | 6 | Compact reference-feature ablation using `mag__time__rms`. |
| Step S | 12 | Full manual feature sets with and without `mag__time__rms`. |
| Step U | 3 | XJTU-SY cross-condition robustness for Step T independent recommended subsets. |
| Total | 27 | First MLP baseline cycle. |

## 3. Main Split / Official Split Conclusions

| Dataset | Task | Recommended Independent Subset | Best Metric Subset | Test Primary | Caveat |
| --- | --- | --- | --- | ---: | --- |
| XJTU-SY | RUL | `full_manual_basic_no_reference` | `full_manual_basic_no_reference` | 0.421645 RMSE | Cross-condition and main split populations differ; cross-condition result is a robustness probe, not same-population improvement. |
| XJTU-SY | HealthState | `compact_non_label_source` | `compact_non_label_source` | 0.371101 WeightedF1 | Pseudo-label task; cross-condition result is strong but not ordinary same-condition bearing generalization. |
| XJTU-SY | EarlyFault | `compact_non_label_source` | `compact_non_label_source` | 0.841682 WeightedF1 | Cross-condition degraded relative to main split; early fault remains condition-sensitive. |
| PHM2012 | RUL | `compact_non_label_source` | `full_manual_basic` | 0.334945 RMSE | Best metric uses label-source feature; independent reporting should use non-reference subset. |
| PHM2012 | HealthState | `compact_non_label_source` | `compact_with_reference` | 0.417337 WeightedF1 | Best metric includes reference feature and task uses pseudo labels; independent reporting should use non-reference subset. |
| PHM2012 | EarlyFault | `compact_non_label_source` | `compact_with_reference` | 0.672350 WeightedF1 | Best metric includes reference feature and task uses pseudo labels; independent reporting should use non-reference subset. |

### RUL

XJTU-SY RUL selects `full_manual_basic_no_reference` as both the best metric subset and the independent recommended subset. PHM2012 RUL has its best metric with `full_manual_basic`, but that subset contains the label-source feature, so the independent recommendation remains `compact_non_label_source` and the full reference result should be reported separately as a sanity or upper comparison.

### HealthState

HealthState favors compact independent subsets for reporting. XJTU-SY selects `compact_non_label_source` as both the independent recommendation and the best metric subset. PHM2012's best metric uses `compact_with_reference`, but because HealthState is derived from the HI/FPT pipeline and the reference subset includes `mag__time__rms`, the independent recommendation remains `compact_non_label_source`.

### EarlyFault

EarlyFault also favors compact independent subsets. XJTU-SY has a main-split tie between compact non-reference and compact reference, so the non-reference subset is the safer reporting baseline. PHM2012's best metric uses `compact_with_reference`; it should be kept as a reference/sanity result, while `compact_non_label_source` remains the independent baseline.

## 4. Compact vs Full

The first-cycle MLP results do not support using full feature sets by default for every task. Full features help XJTU-SY RUL slightly in the independent comparison, but they degrade HealthState and EarlyFault on both datasets. PHM2012 RUL improves most when the full reference subset is allowed, but the independent non-reference comparison still favors compact over full.

Current policy:

- Use `full_manual_basic_no_reference` for XJTU-SY RUL.
- Use `compact_non_label_source` for XJTU-SY HealthState and EarlyFault.
- Use `compact_non_label_source` for PHM2012 independent reporting across RUL, HealthState, and EarlyFault.
- Keep reference/full-reference results as sanity comparisons rather than independent conclusions when they include `mag__time__rms`.

## 5. Cross-Condition Robustness

| Task | Main-Split Comparator Test | Cross-Condition Test | Gap | Interpretation |
| --- | ---: | ---: | ---: | --- |
| RUL | 0.421645 | 0.182463 | -0.239183 | Cross-condition RMSE is lower than the main split reference in this run. |
| HealthState | 0.371101 | 0.702422 | -0.331321 | Cross-condition WeightedF1 is higher than the main split reference in this run. |
| EarlyFault | 0.841682 | 0.754485 | 0.087197 | Cross-condition WeightedF1 is worse than the main split reference. |

The cross-condition check supports keeping the Step T independent recommendations for XJTU-SY, with one caution: EarlyFault drops relative to the main split comparator and should be marked condition-sensitive. RUL and HealthState look strong under this particular train=`35Hz12kN`, val=`37.5Hz11kN`, test=`40Hz10kN` split, but these values are not ordinary same-condition bearing generalization metrics.

## 6. Final Dataset/Task Decisions

| Dataset | Task | Recommended Independent Subset | Best Metric Subset | Final Decision |
| --- | --- | --- | --- | --- |
| XJTU-SY | RUL | `full_manual_basic_no_reference` | `full_manual_basic_no_reference` | keep full_manual_basic_no_reference as independent RUL baseline |
| XJTU-SY | HealthState | `compact_non_label_source` | `compact_non_label_source` | keep compact_non_label_source as independent HealthState baseline |
| XJTU-SY | EarlyFault | `compact_non_label_source` | `compact_non_label_source` | keep compact_non_label_source but mark EarlyFault condition-sensitive |
| PHM2012 | RUL | `compact_non_label_source` | `full_manual_basic` | report compact_non_label_source as independent baseline and full_manual_basic as reference/sanity upper result |
| PHM2012 | HealthState | `compact_non_label_source` | `compact_with_reference` | keep compact_non_label_source as independent HealthState baseline |
| PHM2012 | EarlyFault | `compact_non_label_source` | `compact_with_reference` | keep compact_non_label_source as independent EarlyFault baseline |

The machine-readable version of this table is `baseline_final_decisions.csv`.

## 7. Reference Feature Policy

`mag__time__rms` is a valid diagnostic feature, but it is also the feature used by the HI/FPT labeling pipeline. For derived HealthState and EarlyFault labels, including it can create a label-source shortcut. Therefore:

- Non-reference subsets are the default for independent baseline conclusions.
- Reference subsets are retained as sanity, ablation, or upper-comparison results.
- A reference subset can be reported as the best metric result, but it should not be described as independent evidence.
- PHM2012 RUL, HealthState, and EarlyFault all need this distinction because their best metric subsets include reference features.

## 8. What Is Complete

- Feature analysis for XJTU-SY and PHM2012.
- LaTeX technical documentation for feature concepts and project terminology.
- Baseline planning, feature subsets, metrics, and output conventions.
- Baseline preflight with `mode=inspect_task`.
- 27 real MLP training runs across Step Q, Step R, Step S, and Step U.
- Main split / official split baseline summary.
- XJTU-SY cross-condition robustness check.
- Final baseline decisions for 2 datasets x 3 tasks.

## 9. What Remains

Reasonable next phases are:

1. Tune the MLP baseline with controlled hyperparameter sweeps.
2. Add non-MLP tabular baselines such as linear models, random forest, gradient boosting, or XGBoost-style models if dependencies are acceptable.
3. Run a fuller XJTU-SY cross-condition feature-subset grid if condition robustness becomes the next priority.
4. Add sequence models that consume ordered bearing histories instead of single-row tabular samples.
5. Add physical frequency features such as BPFO, BPFI, BSF, and FTF when reliable shaft speed and bearing geometry metadata are available.

## 10. Decision

- [x] Pass to review.
- [x] Close the first MLP baseline cycle with 27 real training runs and final dataset/task decisions.
- [x] Keep raw checkpoints, prediction parquet files, task manifests, feature tables, labels, HI files, and index files outside the committed report tree.
- [ ] Needs fix.
- [ ] Blocked.

# Step G: XJTU-SY manual_basic vs manual_tsfresh_basic

## 1. Purpose

Compare `manual_basic` and `manual_tsfresh_basic` on the XJTU-SY main bearing-index split for three tasks:

1. RUL
2. Health State
3. Early Fault Detection

This step checks whether tsfresh minimal features add useful information beyond the manual baseline.

## 2. Command Attempted

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_tsfresh_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_all_conditions_3tasks_manual_tsfresh \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

## 3. Config

| Item | Value |
|---|---|
| dataset | xjtu_sy |
| split | xjtu_bearing_index_split |
| feature | manual_tsfresh_basic |
| label | degradation_three_tasks |
| analysis | full_feature_analysis_3tasks |
| run.name | xjtu_all_conditions_3tasks_manual_tsfresh |
| artifact_root | artifacts/feature_analysis |
| feature_source | raw |
| fit_scope | train_only |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/20260619-221616_xjtu_all_conditions_3tasks_manual_tsfresh_99b823be/
```

The run directory contains `config/`, `index/`, `split/`, `run.json`, and `validation_report.json`.
It does not contain `features/`, `labels/`, `hi/`, or `analysis/`, because the process was killed before feature extraction completed.

## 5. Result

| Check | Result | Notes |
|---|---:|---|
| command started | pass | Run directory was created. |
| index built | pass | `index/index_report.json` exists. |
| split built | pass | `split/split_report.json` exists. |
| feature extraction completed | fail | No `features/` directory was produced. |
| labels built | fail | No `labels/` directory was produced. |
| analysis completed | fail | No `analysis/` directory was produced. |
| process exit code | fail | Command exited with code `137`. |

## 6. Root Cause

The failure occurs before analysis, during `manual_tsfresh_basic` feature extraction.

`manual_tsfresh_basic` includes a `tsfresh` backend. The current backend implementation converts every raw sample into one long-format row per time point and channel before calling tsfresh.

For this XJTU-SY main split:

```text
samples: 9216
rows per sample file: 32768
channels used by tsfresh: 2
estimated long-format rows: 9216 * 32768 * 2 = 603,979,776
```

This is too large for the current environment. The process was killed with exit code `137`, which is consistent with system-level termination under resource pressure.

## 7. Comparison with Step F

No valid Step G `feature_ranking.csv` was produced, so there is no trustworthy top-10 comparison yet.

Step F remains the current accepted XJTU-SY main-split result:

```text
feature=manual_basic
num_features=45
num_ranked_features=45
num_leakage_warnings=1
status=done
```

## 8. tsfresh Feature Findings

No `tsfresh__` feature ranking can be reported from this run.

The important finding is architectural/resource-related:

```text
The current tsfresh backend is not suitable for full-size XJTU-SY raw-signal extraction without batching, downsampling, or a more memory-aware extraction path.
```

## 9. Leakage Warnings

No Step G `leakage_report.json` was produced.

The expected caveat still applies once Step G can run: the label config uses HI/FPT source candidates including `mag__time__rms`, `h__time__rms`, and `v__time__rms`, so any feature used as the actual HI source must be marked as label-source and not treated as independent evidence.

## 10. Options Before Rerun

Choose one before rerunning Step G:

1. Add a memory-aware tsfresh backend path that processes samples or batches without constructing one global long-format dataframe.
2. Add explicit downsampling/window compression for tsfresh features and document that method change.
3. Restrict Step G to a smaller diagnostic subset, separate from the official main-split comparison.
4. Skip `manual_tsfresh_basic` for XJTU-SY mainline and keep `manual_basic` as the current feature set.

The current Step G plan explicitly said not to modify the feature backend. Because of that constraint, this report records the failed attempt instead of silently changing the method.

## 11. Decision

- [x] Blocked and deferred: full XJTU-SY `manual_tsfresh_basic` extraction is killed before features are produced.
- [x] Keep `manual_basic` as the XJTU-SY main feature set for the current three-task analysis.
- [x] Defer full-size tsfresh comparison to a later engineering step with batching, downsampling, or windowed extraction.
- [ ] Rerun Step G now.

Next action: proceed to Step H with `manual_basic` condition-wise XJTU-SY analysis.

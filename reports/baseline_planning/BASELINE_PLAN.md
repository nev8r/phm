# Baseline Plan

## 1. Purpose

Step O turns the completed feature-analysis results into an executable baseline experiment plan.

This document is not a training report. It does not contain model scores, checkpoints, predictions, or new experiments. It defines what the next stage should run, why each comparison exists, and how results should be recorded.

## 2. Inputs

Primary inputs:

- `reports/feature_analysis/summary/recommended_features.csv`
- `reports/feature_analysis/summary/final_feature_decisions.md`
- `reports/feature_analysis/FEATURE_ANALYSIS_REPORT.md`
- `reports/feature_analysis/latex/`

The feature-analysis cycle already covered XJTU-SY and PHM2012, three tasks, `manual_basic`, and a limited `manual_tsfresh_basic` comparison. This baseline plan uses those findings as constraints.

## 3. Main Decisions From Feature Analysis

Current mainline feature set:

```text
manual_basic
```

Rationale:

- `manual_basic` runs on both XJTU-SY and PHM2012.
- It provides interpretable amplitude, dispersion, peak-to-peak, shape, and spectral summary features.
- It supports RUL, Health State, and Early Fault tasks.
- XJTU-SY full-size `manual_tsfresh_basic` is blocked by tsfresh long-format memory pressure.
- PHM2012 `manual_tsfresh_basic` runs, but mostly repeats RMS/std/variance/max-style information already captured by manual features.

Reference feature rule:

```text
mag__time__rms is Reference.
```

`mag__time__rms` is the actual HI/FPT source feature in the completed analysis. A model can use it in controlled reference runs, but reports must not describe it as independent evidence for Health State or Early Fault.

Dataset-specific caveats:

- XJTU-SY Early Fault is condition-sensitive. Main-split top features are not enough; cross-condition or condition-wise checks must be reported.
- PHM2012 is more amplitude-dominant under the official split, especially horizontal and magnitude amplitude features.
- Health State and Early Fault are pseudo-label tasks derived from HI/FPT logic, not manually annotated fault-onset labels.

## 4. Tasks

### 4.1 RUL Regression

Target:

```text
piecewise_rul_norm
```

Initial task config:

```text
conf/task/rul_tabular.yaml
```

The first baseline should be tabular, because Step O compares feature subsets rather than sequence architectures. Sequence RUL can be planned after the tabular baseline establishes a reference.

### 4.2 Health State Classification

Target:

```text
health_state_id
```

Initial task config:

```text
conf/task/health_state_tabular.yaml
```

This is a pseudo-health-state classification task derived from HI/FPT. Reports must include class distribution and confusion matrix because class balance may differ by dataset and split.

### 4.3 Early Fault Detection

Target:

```text
early_fault
```

Initial task config:

```text
conf/task/early_fault_tabular.yaml
```

This is a binary pseudo-label task based on FPT. It should be interpreted as early-warning detectability under the current HI/FPT rule, not as a manually annotated physical fault start.

## 5. Datasets and Splits

### 5.1 XJTU-SY

Main split:

```text
xjtu_bearing_index_split
```

Purpose:

- train on bearing suffixes 1-3 across all conditions
- validate on suffix 4
- test on suffix 5
- evaluate baseline behavior under the project main split

Robustness split:

```text
xjtu_cross_condition
```

Purpose:

- verify whether feature-subset conclusions survive condition shift
- especially important for Early Fault

Optional diagnostic split:

```text
xjtu_leave_one_bearing_out
```

Purpose:

- condition-wise diagnosis if a main/cross-condition result is hard to explain
- not part of the first required baseline matrix

### 5.2 PHM2012

Main split:

```text
phm2012_official
```

Purpose:

- follow the project official-split setup
- compare the same feature subsets as XJTU-SY
- check whether horizontal/magnitude amplitude features remain strong in downstream models

## 6. Feature Sets for Baseline

The baseline matrix compares four feature subsets.

### 6.1 `full_manual_basic`

Use all `manual_basic` features.

This is the default strong baseline. It includes `mag__time__rms`, so reports must explicitly mark `label_source_included=yes`.

### 6.2 `full_manual_basic_no_reference`

Use all `manual_basic` features except:

```text
mag__time__rms
```

This tests whether the baseline remains strong without the actual HI/FPT source feature.

### 6.3 `compact_non_label_source`

Use task- and dataset-specific compact features from `FEATURE_SETS.md`, excluding `mag__time__rms`.

This is the main interpretability baseline. If it approaches the full feature set, later work can prefer compact models for explanation and ablation.

### 6.4 `compact_with_reference`

Use `compact_non_label_source` plus:

```text
mag__time__rms
```

This is a sanity/reference run. It estimates the effect of including the HI/FPT source feature. It must be reported separately from independent feature conclusions.

## 7. Baseline Models

First planned model:

```text
model_family = tabular_baseline
model_name = mlp
```

Reason:

- `conf/model/mlp.yaml` exists.
- `ModelFactory` supports `MLP` for tabular regression and classification tasks.
- The current project already has configurable trainer, task datasets, and metrics for the three target types.

Later optional models:

- linear or ridge regression/classification
- random forest
- sequence LSTM/GRU for RUL only

These are not required in Step O because this repository does not yet expose a dedicated sklearn baseline recipe. They should be added only if a later stage explicitly implements them.

## 8. Metrics

Metric definitions are in:

```text
METRICS.md
```

Primary metrics:

- RUL: RMSE
- Health State: Weighted F1
- Early Fault: Weighted F1

Secondary metrics:

- RUL: MAE, MSE, PercentError, optional PHM2012Score if supported by the task config
- Health State: Accuracy, confusion matrix
- Early Fault: Accuracy, confusion matrix, AUC if probability output is available

## 9. Experiment Matrix

The planned matrix is stored in:

```text
EXPERIMENT_MATRIX.csv
```

Required first pass:

- XJTU-SY main split, 3 tasks, 4 feature subsets
- PHM2012 official split, 3 tasks, 4 feature subsets

Optional phase-2 robustness:

- XJTU-SY cross-condition split, 3 tasks, 4 feature subsets

## 10. Output Convention

Output rules are in:

```text
OUTPUT_CONVENTION.md
```

Later training runs should write raw run artifacts outside Git by default. Curated reports may be copied into `reports/baseline_results/<experiment_id>/`.

## 11. Acceptance Criteria for Next Stage

The next stage can start training only after Step O is accepted.

Minimum acceptance criteria:

- every required experiment has a unique `experiment_id`
- every experiment records dataset, split, task, feature subset, label config, model, metrics, and required flag
- `mag__time__rms` inclusion is explicit through the feature subset
- XJTU-SY cross-condition runs are present as optional robustness checks
- no checkpoints, predictions, or model results are committed as part of Step O

## 12. Next Stage

After Step O is accepted, the next stage should implement or run the first required tabular MLP baseline matrix. It should not mix result generation with new feature-analysis changes.

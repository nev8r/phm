# Feature Sets

This file translates the feature-analysis recommendations into baseline feature-subset definitions.

The source table is:

```text
reports/feature_analysis/summary/recommended_features.csv
```

## 1. Naming Rules

Feature subset names used by `EXPERIMENT_MATRIX.csv`:

- `full_manual_basic`
- `full_manual_basic_no_reference`
- `compact_non_label_source`
- `compact_with_reference`

`manual_basic` contains 45 features in the completed feature-analysis runs.

## 2. Reference Feature

Reference feature:

```text
mag__time__rms
```

Reason:

- It is the actual HI/FPT source feature in the completed analyses.
- Health State and Early Fault pseudo-labels depend on the HI/FPT construction.
- Strong model performance with this feature included may partly reflect label construction.

Reporting rule:

- Runs including this feature must report `label_source_included=yes`.
- Runs excluding this feature must report `label_source_included=no`.
- Reference runs can be used for sanity checks and ablation, but not as independent feature evidence.

## 3. `full_manual_basic`

Definition:

```text
manual_basic all features
```

Includes:

```text
mag__time__rms
```

Purpose:

- default strong baseline
- upper reference for the current feature pipeline
- comparison target for compact subsets

## 4. `full_manual_basic_no_reference`

Definition:

```text
manual_basic - mag__time__rms
```

Purpose:

- test whether models remain useful without the actual label-source feature
- isolate independent amplitude, dispersion, shape, and spectral features

## 5. `compact_non_label_source`

Definition:

```text
task-specific compact subset from recommended_features.csv, excluding mag__time__rms
```

Purpose:

- interpretable compact baseline
- lower-dimensional ablation against `full_manual_basic_no_reference`
- preferred subset if performance loss is small

### 5.1 XJTU-SY RUL

```text
mag__time__mean
mag__time__mean_abs
mag__time__std
h__time__mean_abs
h__time__std
v__time__mean_abs
v__time__std
```

Notes:

- magnitude features are the safest global family
- horizontal and vertical features are useful secondary support
- vertical features should be interpreted with split-aware validation

### 5.2 XJTU-SY Health State

```text
mag__time__mean
mag__time__mean_abs
mag__time__std
h__time__mean_abs
h__time__std
h__time__rms
```

Notes:

- magnitude features are more robust under cross-condition checks
- horizontal features are strong in the main split but should be treated as secondary

### 5.3 XJTU-SY Early Fault

```text
mag__time__mean
mag__time__mean_abs
mag__time__std
v__time__std
v__time__mean_abs
```

Notes:

- XJTU-SY Early Fault is condition-sensitive
- horizontal and spectral C-level features may be used for diagnostics, but are not part of the default compact subset

### 5.4 PHM2012 RUL

```text
h__time__mean_abs
mag__time__mean
mag__time__mean_abs
h__time__rms
h__time__std
v__time__mean_abs
mag__time__std
```

Notes:

- horizontal amplitude is strongest
- magnitude amplitude is also stable
- bearing-specific variance should be reported

### 5.5 PHM2012 Health State

```text
h__time__mean_abs
h__time__std
h__time__rms
mag__time__mean
mag__time__mean_abs
```

Notes:

- horizontal amplitude and dispersion are the main family
- magnitude features are secondary support

### 5.6 PHM2012 Early Fault

```text
h__time__mean_abs
mag__time__mean
mag__time__mean_abs
h__time__std
h__time__rms
v__time__mean_abs
v__time__std
```

Notes:

- amplitude features dominate
- vertical features are secondary support
- spectral frequency features remain diagnostic only

## 6. `compact_with_reference`

Definition:

```text
compact_non_label_source + mag__time__rms
```

Purpose:

- estimate how much the HI/FPT source feature changes baseline performance
- verify label-source sanity
- report separately from independent compact baseline conclusions

## 7. Implementation Notes for Later Stages

The current repository does not yet define a feature-subset config layer for these named subsets.

Later implementation can choose one of two paths:

- add a task/data filtering config that selects columns by subset name
- generate curated feature parquet/csv views for each subset before building task datasets

Either path must keep train-only cleaning and split boundaries intact.

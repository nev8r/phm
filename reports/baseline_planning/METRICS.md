# Metrics

This file defines the planned baseline metrics for the three downstream tasks.

Step O does not compute these metrics. It defines what later training stages must report.

## 1. RUL Regression

Target:

```text
piecewise_rul_norm
```

Primary metric:

- RMSE

Secondary metrics:

- MAE
- MSE
- PercentError
- PHM2012Score, if supported by the resolved task config

Interpretation notes:

- If the target is `piecewise_rul_norm`, RMSE/MAE are normalized RUL errors.
- If a later task outputs time steps, seconds, or cycles, the unit must be stated in the experiment report.
- Report validation and test metrics separately.
- For PHM2012, note whether any PHM-style asymmetric score is normalized or raw.

Minimum reporting table:

| Metric | Val | Test | Unit |
| --- | ---: | ---: | --- |
| RMSE | | | normalized RUL |
| MAE | | | normalized RUL |
| MSE | | | normalized RUL squared |

## 2. Health State Classification

Target:

```text
health_state_id
```

Primary metric:

- Weighted F1

Secondary metrics:

- Accuracy
- macro F1, if available
- confusion matrix
- per-class support

Reason for Weighted F1:

Health-state classes may be imbalanced because pseudo states are derived from HI/FPT thresholds and bearing life lengths. Accuracy can hide poor minority-class behavior.

Minimum reporting table:

| Metric | Val | Test |
| --- | ---: | ---: |
| Weighted F1 | | |
| Accuracy | | |
| Macro F1 | | |

The confusion matrix should be included as a figure or compact table in curated reports.

## 3. Early Fault Detection

Target:

```text
early_fault
```

Primary metric:

- Weighted F1

Secondary metrics:

- Accuracy
- confusion matrix
- AUC, if the model/reporting path exposes probabilities or calibrated scores
- per-class support

Interpretation notes:

- Early Fault is an FPT pseudo-label task.
- It is not a manually annotated physical fault-onset label.
- XJTU-SY Early Fault must report the split used, because condition sensitivity is part of the feature-analysis conclusion.

Minimum reporting table:

| Metric | Val | Test |
| --- | ---: | ---: |
| Weighted F1 | | |
| Accuracy | | |
| AUC | | |

## 4. Cross-Experiment Reporting Rules

Every baseline report should include:

- dataset
- split
- task
- feature subset
- whether `mag__time__rms` is included
- primary metric
- secondary metrics
- validation/test difference
- caveats from feature analysis

Do not compare reference-including runs directly against non-reference runs without marking the label-source caveat.

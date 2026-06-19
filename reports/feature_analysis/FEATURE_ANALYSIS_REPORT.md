# Bearing Feature Analysis Report

## 1. Executive Summary

This report summarizes the completed feature-analysis cycle for XJTU-SY and PHM2012.

Final mainline decision:

```text
Use manual_basic for downstream baseline planning.
```

The strongest cross-dataset pattern is amplitude or energy-like time-domain features. These features support RUL, Health State, and Early Fault Detection across both datasets. `mag__time__rms` is retained only as a label-source reference because it is the actual HI/FPT source feature in the completed analyses.

tsfresh decision:

- XJTU-SY `manual_tsfresh_basic` is blocked and deferred because full-size extraction exceeds the current backend's practical memory path.
- PHM2012 `manual_tsfresh_basic` succeeds, but useful `tsfresh__` features mostly duplicate manual RMS, standard deviation, variance, max, and absolute max statistics.
- Therefore tsfresh is not adopted as the current mainline feature set.

## 2. Analysis Setup

Datasets:

- XJTU-SY
- PHM2012

Tasks:

- RUL
- Health State
- Early Fault Detection

Feature sets:

- `manual_basic`
- `manual_tsfresh_basic`

Label config:

```text
degradation_three_tasks
```

Analysis config:

```text
full_feature_analysis_3tasks
```

Ranking rule:

```text
fit_scope = train_only
```

This means feature rankings are computed from the train split only. Validation and test splits are used for distribution checks and visual review.

## 3. XJTU-SY Results

### Main Split Result

Step F analyzed XJTU-SY `manual_basic` on the main all-condition bearing-index split.

Main findings:

- RUL is dominated by magnitude and energy-like time-domain features.
- HealthState is strongly supported by horizontal amplitude features.
- EarlyFault is supported by horizontal amplitude features in the main split, but this claim needed stability checks.
- `mag__time__rms` is the actual HI/FPT label-source feature.

### Condition-Wise Stability

Step H showed:

- RUL amplitude features are stable across the three operating conditions.
- HealthState is mostly amplitude-driven, but channel preference changes.
- EarlyFault is the most condition-sensitive task.

Condition-specific EarlyFault behavior:

- C1: spectral entropy signals are strong.
- C2: peak-to-peak shock-like features are strong.
- C3: horizontal amplitude features are strong.

### Cross-Condition Robustness

Step I used:

- train: `35Hz12kN`
- val: `37.5Hz11kN`
- test: `40Hz10kN`

Main findings:

- RUL amplitude features mostly survive cross-condition analysis.
- HealthState is safer with magnitude amplitude features than with pure horizontal features.
- EarlyFault horizontal amplitude features should be downgraded to condition-sensitive claims.
- Spectral entropy in Step I mostly reflects Condition 1 behavior.

### XJTU-SY Feature Set Decision

Use `manual_basic`.

Do not use full-size `manual_tsfresh_basic` in the current mainline:

- Step G was killed before feature extraction completed.
- The current tsfresh backend would construct about 604M long-format rows.
- tsfresh comparison is deferred to a later engineering stage.

## 4. PHM2012 Results

### manual_basic Result

Step J analyzed PHM2012 `manual_basic` on the official split.

Main findings:

- RUL is dominated by horizontal and magnitude amplitude features.
- HealthState is strongly horizontal-channel amplitude-driven.
- EarlyFault is also dominated by horizontal and magnitude amplitude features.
- Spectral features are mostly secondary.
- `mag__time__rms` is again the actual HI/FPT label-source feature.

### manual_tsfresh_basic Comparison

Step K successfully ran PHM2012 `manual_tsfresh_basic`.

Main findings:

- The ranking contains 20 `tsfresh__` features.
- `tsfresh__` features enter top-10 lists for RUL, HealthState, and EarlyFault.
- The strongest `tsfresh__` entries mainly duplicate manual RMS, standard deviation, variance, max, and absolute max features.
- This supports the amplitude story but does not justify replacing `manual_basic`.

### PHM2012 Feature Set Decision

Use `manual_basic`.

Do not adopt `manual_tsfresh_basic` as the current mainline because it is more expensive and mostly redundant.

## 5. Cross-Dataset Comparison

Common pattern:

- Amplitude and energy-like time-domain features are consistently useful.

Dataset differences:

- XJTU-SY EarlyFault is more condition-sensitive.
- PHM2012 is more consistently horizontal and magnitude amplitude-driven.
- XJTU-SY full-size tsfresh is blocked, while PHM2012 tsfresh succeeds but does not add a new feature family.

## 6. Final Recommended Features

The machine-readable recommendation table is:

```text
reports/feature_analysis/summary/recommended_features.csv
```

High-level recommendations:

| Dataset | Task | Main Recommendation | Reference |
|---|---|---|---|
| XJTU-SY | RUL | magnitude amplitude and dispersion features | `mag__time__rms` |
| XJTU-SY | HealthState | magnitude amplitude plus horizontal secondary features | `mag__time__rms` |
| XJTU-SY | EarlyFault | magnitude amplitude with condition-sensitive auxiliaries | `mag__time__rms` |
| PHM2012 | RUL | horizontal and magnitude amplitude features | `mag__time__rms` |
| PHM2012 | HealthState | horizontal amplitude features | `mag__time__rms` |
| PHM2012 | EarlyFault | horizontal and magnitude amplitude features | `mag__time__rms` |

## 7. Limitations

- HealthState and EarlyFault labels are pseudo labels derived from HI/FPT logic.
- `mag__time__rms` is a label-source reference and should not be overclaimed.
- XJTU-SY EarlyFault is condition-sensitive.
- PHM2012 `manual_tsfresh_basic` succeeded, but its useful features mostly duplicate manual statistics.
- XJTU-SY full-size tsfresh requires a memory-aware extraction path before it can be compared fairly.
- Bearing-characteristic-frequency features such as BPFO, BPFI, BSF, and FTF were not engineered in this cycle.
- No downstream model training is included in this feature-analysis report.

## 8. Next Step

Proceed to baseline planning with:

```text
feature_set = manual_basic
recommendation_source = reports/feature_analysis/summary/recommended_features.csv
```

Baseline planning should define separate experiments for:

- RUL regression
- Health State classification
- Early Fault detection

# XJTU-SY Feature Summary

## 1. Scope

- Dataset: XJTU-SY bearing run-to-failure dataset.
- Tasks: RUL, Health State, Early Fault Detection.
- Main split: `xjtu_bearing_index_split`.
- Stability checks: condition-wise leave-one-bearing-out and cross-condition split.
- Main feature set: `manual_basic`.
- Label config: `degradation_three_tasks`.
- Analysis config: `full_feature_analysis_3tasks`.

No new analysis was run for this summary. It consolidates Step F, Step G, Step H, and Step I.

## 2. Main Findings

### RUL

The most reliable XJTU-SY RUL family is amplitude or energy-like time-domain features.

Strong candidates:

- `mag__time__mean`
- `mag__time__mean_abs`
- `mag__time__std`
- `h__time__mean_abs`
- `h__time__rms`
- `h__time__std`
- `v__time__mean_abs`
- `v__time__std`

Evidence:

- Step F ranked magnitude and horizontal amplitude features at the top of the main split.
- Step H showed the same amplitude family appears in the top-10 across all three operating conditions.
- Step I kept magnitude and vertical amplitude features strong under cross-condition train-only ranking.

Reference feature:

- `mag__time__rms` is the actual HI/FPT label-source feature. It is useful as a sanity reference but should not be treated as independent predictive evidence.

### Health State

Health State is also amplitude-driven. The strongest main-split features are horizontal amplitude features, but cross-condition analysis suggests magnitude features are more stable than pure horizontal features.

Recommended candidates:

- A-level: `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std`
- B-level: `h__time__mean_abs`, `h__time__std`, `h__time__rms`
- C-level diagnostic: `v__spectral__entropy` for Condition 1 style behavior
- Reference: `mag__time__rms`

Evidence:

- Step F strongly favored `h__time__mean_abs`, `h__time__rms`, and `h__time__std`.
- Step H found `h__time__mean_abs` most stable, while `h__time__rms` and `h__time__std` were stronger in C2/C3.
- Step I downgraded pure horizontal HealthState claims and favored magnitude amplitude features for cross-condition robustness.

### Early Fault

Early Fault is the most condition-sensitive XJTU-SY task.

Recommended candidates:

- A-level: `mag__time__mean`, `mag__time__mean_abs`
- B-level: `mag__time__std`, `v__time__std`, `v__time__mean_abs`
- C-level condition-sensitive: `h__time__mean_abs`, `h__time__std`, `h__time__rms`, `v__spectral__entropy`
- Reference: `mag__time__rms`

Evidence:

- Step F favored horizontal amplitude features.
- Step H showed condition-specific EarlyFault behavior: C1 favored spectral entropy, C2 favored peak-to-peak shock-like features, and C3 favored horizontal amplitude.
- Step I downgraded global horizontal EarlyFault claims and kept magnitude amplitude features as the safer mainline.

## 3. Stability Checks

### Condition-Wise

Step H validated the main finding under the three XJTU-SY operating conditions:

- RUL amplitude features are stable.
- HealthState is mostly stable but channel preference changes by condition.
- EarlyFault is condition-sensitive and should be interpreted carefully.

### Cross-Condition

Step I used:

- train: `35Hz12kN`
- val: `37.5Hz11kN`
- test: `40Hz10kN`

The ranking is train-only, so the top features reflect the train condition. Val/test were used for distribution and visualization checks. This step reinforced amplitude features for RUL, favored magnitude features for HealthState, and showed that EarlyFault needs condition-aware caveats.

## 4. Feature Set Decision

- `manual_basic`: accepted as the current XJTU-SY mainline feature set.
- `manual_tsfresh_basic`: blocked and deferred for full XJTU-SY because the current tsfresh backend would construct about 604M long-format rows and was killed before feature extraction completed.

## 5. Final XJTU-SY Recommended Features

| Task | A-level | B-level | C-level | Reference |
|---|---|---|---|---|
| RUL | `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std` | `h__time__mean_abs`, `h__time__rms`, `h__time__std`, `v__time__mean_abs`, `v__time__std` | none | `mag__time__rms` |
| HealthState | `mag__time__mean`, `mag__time__mean_abs`, `mag__time__std` | `h__time__mean_abs`, `h__time__std`, `h__time__rms` | `v__spectral__entropy` | `mag__time__rms` |
| EarlyFault | `mag__time__mean`, `mag__time__mean_abs` | `mag__time__std`, `v__time__std`, `v__time__mean_abs` | `h__time__mean_abs`, `h__time__std`, `h__time__rms`, `v__spectral__entropy` | `mag__time__rms` |

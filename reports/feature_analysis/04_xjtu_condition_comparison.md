# Step H: XJTU-SY Condition-wise manual_basic Analysis

## 1. Purpose

Validate whether the Step F main-split `manual_basic` findings are stable within each XJTU-SY operating condition.

Step H analyzes:

1. 35Hz12kN
2. 37.5Hz11kN
3. 40Hz10kN

It uses the same three-task setup:

- RUL
- Health State
- Early Fault Detection

Step H does not run tsfresh, PHM2012, cross-condition analysis, or model training.

## 2. Commands

### H1. Condition 1

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_c1_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=35Hz12kN \
  split.test_bearing_id=Bearing1_5 \
  split.val_bearing_id=Bearing1_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-225611_xjtu_c1_3tasks_manual_basic_7a08cc4f`
- Report directory: `reports/feature_analysis/xjtu_sy/condition_1_manual_basic`

### H2. Condition 2

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_c2_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=37.5Hz11kN \
  split.test_bearing_id=Bearing2_5 \
  split.val_bearing_id=Bearing2_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-225749_xjtu_c2_3tasks_manual_basic_51d6cc04`
- Report directory: `reports/feature_analysis/xjtu_sy/condition_2_manual_basic`

### H3. Condition 3

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_leave_one_bearing_out \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_c3_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu \
  split.condition_id=40Hz10kN \
  split.test_bearing_id=Bearing3_5 \
  split.val_bearing_id=Bearing3_4
```

- Run directory: `artifacts/feature_analysis/runs/20260619-225926_xjtu_c3_3tasks_manual_basic_6304e227`
- Report directory: `reports/feature_analysis/xjtu_sy/condition_3_manual_basic`

## 3. Sanity Checks

| Condition | analysis_report.ok | feature_source | fit_scope | num_features | num_ranked_features | leakage warnings | Status |
|---|---:|---|---|---:|---:|---:|---|
| C1 35Hz12kN | true | raw | train_only | 45 | 45 | 1 | pass |
| C2 37.5Hz11kN | true | raw | train_only | 45 | 45 | 1 | pass |
| C3 40Hz10kN | true | raw | train_only | 45 | 45 | 1 | pass |

All three `feature_summary.csv` files report 0 missing values, 0 NaN values, 0 Inf values, and no constant feature rows.

## 4. Leakage Summary

| Condition | Actual label-source feature | Warning text |
|---|---|---|
| C1 | `mag__time__rms` | Feature was used as HI source for FPT-based labels. |
| C2 | `mag__time__rms` | Feature was used as HI source for FPT-based labels. |
| C3 | `mag__time__rms` | Feature was used as HI source for FPT-based labels. |

The actual HI/FPT source is consistent with Step F: `mag__time__rms`. It should remain a label-source reference feature, not independent evidence for Health State or Early Fault.

## 5. RUL Stability

| Feature | C1 Rank | C2 Rank | C3 Rank | Top10 Count | Interpretation |
|---|---:|---:|---:|---:|---|
| `mag__time__mean` | 1 | 6 | 4 | 3 | Stable RUL candidate. |
| `mag__time__mean_abs` | 1 | 6 | 4 | 3 | Stable RUL candidate, redundant with magnitude mean in these runs. |
| `mag__time__rms` | 4 | 2 | 6 | 3 | Stable but label-source caveat applies. |
| `h__time__rms` | 8 | 3 | 2 | 3 | Stable RUL candidate. |
| `h__time__mean_abs` | 10 | 4 | 1 | 3 | Stable RUL candidate and aligns with Step F Health/EarlyFault findings. |
| `h__time__std` | 9 | 5 | 3 | 3 | Stable RUL candidate, close to RMS. |
| `mag__time__std` | 7 | 1 | 10 | 3 | Stable RUL candidate. |
| `v__time__mean_abs` | 3 | 9 | 7 | 3 | Stable candidate, more vertical-channel sensitive. |

Discussion:

- Step F's energy-like RUL conclusion is stable across all three conditions.
- `mag__time__rms` is stable but remains a label-source feature.
- Magnitude and horizontal time-domain amplitude features are the most robust RUL candidates.

## 6. Health State Stability

| Feature | C1 Rank | C2 Rank | C3 Rank | Top10 Count | Interpretation |
|---|---:|---:|---:|---:|---|
| `h__time__mean_abs` | 9 | 2 | 1 | 3 | Stable HealthState candidate. |
| `mag__time__mean` | 3 | 9 | 4 | 3 | Stable but less specific than horizontal features. |
| `mag__time__mean_abs` | 3 | 9 | 4 | 3 | Stable, redundant with magnitude mean. |
| `mag__time__std` | 8 | 1 | 9 | 3 | Stable HealthState candidate. |
| `mag__time__rms` | 7 | 6 | 6 | 3 | Stable but label-source caveat applies. |
| `h__time__rms` | 11 | 3 | 2 | 2 | Strong in C2/C3; weaker in C1. |
| `h__time__std` | 12 | 4 | 3 | 2 | Strong in C2/C3; weaker in C1. |
| `v__time__mean_abs` | 2 | 13 | 7 | 2 | Condition-sensitive, especially strong in C1/C3. |
| `h__time__ptp` | 14 | 5 | 8 | 2 | Stable enough as a secondary candidate. |

Discussion:

- Step F's horizontal amplitude HealthState conclusion is supported mainly by C2 and C3.
- C1 includes stronger vertical and spectral entropy signals, so horizontal RMS/std are not universally top-10 there.
- `h__time__mean_abs` is the most stable Step F HealthState feature across all three conditions.

## 7. Early Fault Stability

| Feature | C1 Rank | C2 Rank | C3 Rank | Top10 Count | Interpretation |
|---|---:|---:|---:|---:|---|
| `mag__time__mean` | 7 | 6 | 4 | 3 | Stable EarlyFault candidate. |
| `mag__time__mean_abs` | 7 | 6 | 4 | 3 | Stable, redundant with magnitude mean. |
| `mag__time__rms` | 9 | 5 | 6 | 3 | Stable but label-source caveat applies. |
| `v__time__std` | 10 | 8 | 10 | 3 | Stable vertical dispersion candidate. |
| `mag__time__std` | 12 | 4 | 9 | 2 | Stable enough as secondary candidate. |
| `v__time__mean_abs` | 13 | 10 | 8 | 2 | Condition-sensitive but appears in C2/C3 top-10. |
| `h__time__mean_abs` | 15 | 13 | 1 | 1 | Strong in C3, not stable across C1/C2. |
| `h__time__std` | 17 | 11 | 2 | 1 | Strong in C3, not stable across C1/C2. |
| `h__time__rms` | 16 | 12 | 3 | 1 | Strong in C3, not stable across C1/C2. |
| `mag__time__ptp` | 19 | 1 | 14 | 1 | Condition-specific C2 shock candidate. |
| `v__time__ptp` | 14 | 2 | 15 | 1 | Condition-specific C2 shock candidate. |
| `v__spectral__entropy` | 1 | 16 | 24 | 1 | Condition-specific C1 spectral candidate. |

Discussion:

- EarlyFault is the most condition-sensitive task in Step H.
- C1 favors spectral entropy features, C2 favors peak-to-peak shock features, and C3 favors horizontal amplitude features.
- Step F's horizontal amplitude EarlyFault conclusion is strong in C3 but should be downgraded from "globally stable" to "important but condition-sensitive."

## 8. Figures Reviewed

For each condition, these main figures were copied and reviewed:

- `figures/rul_top_features.png`
- `figures/degradation_score_heatmap.png`
- `figures/health_state_boxplots.png`
- `figures/early_fault_effects.png`
- `figures/feature_recommendation_matrix.png`
- `figures/feature_score_heatmap.png`

Selected curves were copied under each condition's `figures/curves/` directory. The file `selected_curves.txt` records copied and missing requested curves.

Figure review summary:

- RUL top-feature charts are dominated by amplitude or energy-like features in all three conditions.
- Health-state boxplots show strongest separation in C2/C3; C1 has more vertical/spectral emphasis.
- EarlyFault plots show different condition-specific patterns: spectral entropy in C1, peak-to-peak shock features in C2, and horizontal amplitude features in C3.
- All copied PNG files were checked as nonblank.

## 9. Issues / Warnings

- Condition imbalance: C1 has 616 samples, C2 has 1566 samples, and C3 has 7034 samples. C1 and C2 conclusions should be treated as less stable than C3.
- Val/test size issue: Step D recorded C2 validation size as 42 and C1 test size as 52; small splits may make visual separation look brittle.
- Leakage: all three condition runs use `mag__time__rms` as the actual HI/FPT label-source feature.
- Plot quality: selected aggregate curves are usable. Some requested C1 curves were not generated by the plotting utility and are listed in `condition_1_manual_basic/figures/curves/selected_curves.txt`.
- Other: condition-wise analysis supports `manual_basic` as the current XJTU-SY mainline feature set.

## 10. Decision

- [ ] Pass
- [x] Needs review
- [ ] Needs rerun
- [ ] Blocked

Next action: review condition-wise stability findings. After acceptance, proceed to Step I, XJTU-SY cross-condition `manual_basic` robustness analysis.

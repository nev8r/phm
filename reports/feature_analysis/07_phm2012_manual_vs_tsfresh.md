# Step K: PHM2012 manual_basic vs manual_tsfresh_basic

## 1. Purpose

Compare `manual_basic` and `manual_tsfresh_basic` on the PHM2012 official split for three tasks:

1. RUL
2. Health State
3. Early Fault Detection

This step checks whether tsfresh minimal features add useful information beyond the manual baseline.

Step K does not run XJTU-SY, does not train models, does not change the tsfresh backend, and does not use a subset.

## 2. Command

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_tsfresh_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=phm2012_3tasks_manual_tsfresh \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/phm2012
```

## 3. Config

| Item | Value |
|---|---|
| dataset | phm2012 |
| split | phm2012_official |
| feature | manual_tsfresh_basic |
| label | degradation_three_tasks |
| analysis | full_feature_analysis_3tasks |
| run.name | phm2012_3tasks_manual_tsfresh |
| artifact_root | artifacts/feature_analysis |
| feature_source | raw |
| fit_scope | train_only |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/20260619-235620_phm2012_3tasks_manual_tsfresh_c5ccee59/
```

## 5. Files Copied

```text
reports/feature_analysis/phm2012/manual_tsfresh_basic/
```

Copied analysis files:

```text
command.txt
analysis_report.json
leakage_report.json
feature_summary.csv
rul_correlation.csv
degradation_scores.csv
health_state_separability.csv
early_fault_scores.csv
feature_ranking.csv
feature_cards.csv
feature_recommendations.md
```

Copied figures:

```text
figures/rul_top_features.png
figures/degradation_score_heatmap.png
figures/health_state_boxplots.png
figures/early_fault_effects.png
figures/feature_recommendation_matrix.png
figures/feature_score_heatmap.png
```

Selected curves were copied under:

```text
reports/feature_analysis/phm2012/manual_tsfresh_basic/figures/curves/
```

`selected_curves.txt` records copied and missing requested curves.

## 6. Sanity Checks

| Check | Result | Notes |
|---|---:|---|
| `analysis_report.ok` | pass | `true` |
| `analysis_name=full_feature_analysis_3tasks` | pass | Matches Step K config. |
| `feature_source=raw` | pass | Analysis used raw manual+tsfresh features. |
| `fit_scope=train_only` | pass | Scores are fit on the train split only. |
| `num_features > 0` | pass | 44 raw features. |
| `num_ranked_features > 0` | pass | 44 ranked features. |
| `feature_ranking exists` | pass | CSV copied to report directory. |
| `tsfresh__ features exist` | pass | 20 `tsfresh__` features in ranking. |
| `leakage_report checked` | pass | One HI/FPT label-source warning found. |
| `figures exist` | pass | Required figures and selected available curves exist. |

Feature extraction report:

| Item | Value |
|---|---:|
| raw features | 44 |
| cleaned features | 42 |
| dropped features | 2 |

Dropped features:

```text
tsfresh__h__length
tsfresh__v__length
```

These two length features are constant because each PHM2012 snapshot has the same row count. They are not useful feature candidates.

## 7. Leakage Summary

| Actual label-source feature | Warning |
|---|---|
| `mag__time__rms` | Feature was used as HI source for FPT-based labels. |

`mag__time__rms` remains a label-source reference feature and should not be used as independent evidence for Health State or Early Fault claims.

## 8. Top-10 Comparison with Step J

Detailed comparison is stored in:

```text
reports/feature_analysis/summary/phm2012_manual_vs_tsfresh_top10.csv
```

### RUL

| Item | Result |
|---|---|
| `manual_basic` top10 | `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`, `h__time__rms`, `h__time__std`, `mag__time__rms`, `v__time__mean_abs`, `mag__time__std`, `v__time__std`, `v__time__rms` |
| `manual_tsfresh_basic` top10 | `mag__time__mean`, `h__time__rms`, `tsfresh__h__root_mean_square`, `h__time__std`, `tsfresh__h__standard_deviation`, `tsfresh__h__variance`, `mag__time__rms`, `tsfresh__v__variance`, `mag__time__std`, `tsfresh__v__standard_deviation` |
| overlap | `mag__time__mean`, `h__time__rms`, `h__time__std`, `mag__time__rms`, `mag__time__std` |
| new manual_tsfresh features | `tsfresh__h__root_mean_square`, `tsfresh__h__standard_deviation`, `tsfresh__h__variance`, `tsfresh__v__variance`, `tsfresh__v__standard_deviation` |
| `tsfresh__` features in top10 | 5 |

### Health State

| Item | Result |
|---|---|
| `manual_basic` top10 | `h__time__mean_abs`, `h__time__std`, `h__time__rms`, `mag__time__mean`, `mag__time__mean_abs`, `mag__time__rms`, `mag__spectral__entropy`, `h__time__ptp`, `mag__time__std`, `mag__spectral__centroid` |
| `manual_tsfresh_basic` top10 | `h__time__std`, `tsfresh__h__standard_deviation`, `h__time__rms`, `tsfresh__h__root_mean_square`, `mag__time__mean`, `mag__time__rms`, `mag__spectral__entropy`, `tsfresh__h__maximum`, `mag__time__std`, `tsfresh__h__absolute_maximum` |
| overlap | `h__time__std`, `h__time__rms`, `mag__time__mean`, `mag__time__rms`, `mag__spectral__entropy`, `mag__time__std` |
| new manual_tsfresh features | `tsfresh__h__standard_deviation`, `tsfresh__h__root_mean_square`, `tsfresh__h__maximum`, `tsfresh__h__absolute_maximum` |
| `tsfresh__` features in top10 | 4 |

### Early Fault

| Item | Result |
|---|---|
| `manual_basic` top10 | `h__time__mean_abs`, `mag__time__mean`, `mag__time__mean_abs`, `h__time__std`, `h__time__rms`, `mag__time__rms`, `v__time__mean_abs`, `v__time__rms`, `v__time__std`, `mag__time__std` |
| `manual_tsfresh_basic` top10 | `mag__time__mean`, `h__time__std`, `tsfresh__h__standard_deviation`, `h__time__rms`, `tsfresh__h__root_mean_square`, `mag__time__rms`, `tsfresh__v__standard_deviation`, `v__time__std`, `v__time__rms`, `tsfresh__v__root_mean_square` |
| overlap | `mag__time__mean`, `h__time__std`, `h__time__rms`, `mag__time__rms`, `v__time__std`, `v__time__rms` |
| new manual_tsfresh features | `tsfresh__h__standard_deviation`, `tsfresh__h__root_mean_square`, `tsfresh__v__standard_deviation`, `tsfresh__v__root_mean_square` |
| `tsfresh__` features in top10 | 4 |

## 9. tsfresh Feature Findings

| Task | `tsfresh__` features in top10 | Interpretation |
|---|---:|---|
| RUL | 5 | Mostly duplicates manual RMS/std/variance amplitude information. |
| Health State | 4 | Horizontal standard deviation, RMS, max, and absolute max confirm the same horizontal amplitude story. |
| Early Fault | 4 | Horizontal and vertical tsfresh RMS/std mirror manual amplitude features. |

Important examples:

| Feature | RUL Rank | Health Rank | EarlyFault Rank | Interpretation |
|---|---:|---:|---:|---|
| `tsfresh__h__standard_deviation` | 4 | 1 | 2 | Equivalent to horizontal standard deviation; useful but redundant. |
| `tsfresh__h__root_mean_square` | 2 | 3 | 4 | Equivalent to horizontal RMS; useful but redundant. |
| `tsfresh__h__variance` | 6 | 13 | 15 | Adds a variance transform, but not a new feature family. |
| `tsfresh__h__maximum` | 14 | 8 | 17 | Secondary horizontal amplitude feature. |
| `tsfresh__v__standard_deviation` | 10 | 17 | 7 | Confirms vertical dispersion for EarlyFault. |
| `tsfresh__v__root_mean_square` | 12 | 19 | 9 | Confirms vertical RMS for EarlyFault. |

Step K shows that PHM2012 can run full-size `manual_tsfresh_basic`, unlike XJTU-SY Step G. However, the useful `tsfresh__` entries mostly repeat amplitude/statistical features that `manual_basic` already provides.

## 10. Figures Reviewed

Reviewed required figures:

- `figures/rul_top_features.png`
- `figures/degradation_score_heatmap.png`
- `figures/health_state_boxplots.png`
- `figures/early_fault_effects.png`
- `figures/feature_recommendation_matrix.png`
- `figures/feature_score_heatmap.png`

Copied selected aggregate curves:

- `figures/curves/mag__time__mean.png`
- `figures/curves/h__time__rms.png`
- `figures/curves/tsfresh__h__root_mean_square.png`
- `figures/curves/h__time__std.png`
- `figures/curves/tsfresh__h__standard_deviation.png`
- `figures/curves/tsfresh__h__variance.png`
- `figures/curves/tsfresh__v__standard_deviation.png`
- `figures/curves/tsfresh__h__maximum.png`

Requested but not generated by the plotting utility:

- `tsfresh__v__variance`
- `tsfresh__h__absolute_maximum`
- `tsfresh__v__root_mean_square`

All copied PNG files were checked as nonblank.

## 11. Issues / Warnings

- Redundancy: top `tsfresh__` features mostly duplicate manual RMS/std or simple amplitude statistics.
- Constant features: `tsfresh__h__length` and `tsfresh__v__length` are constant and dropped by the cleaner.
- Leakage: `mag__time__rms` is still the actual HI/FPT label-source feature.
- Plot coverage: several requested top10 tsfresh curves were not generated by the plotting utility and are recorded in `selected_curves.txt`.
- Scope: this success applies to PHM2012; it does not invalidate Step G's XJTU-SY full-size tsfresh blockage.

## 12. Decision

- [ ] `manual_tsfresh_basic` adds useful features and should be considered for later baselines.
- [x] `manual_tsfresh_basic` does not add clear value; keep `manual_basic` as PHM2012 main feature set.
- [ ] Needs more evidence before deciding.
- [ ] Blocked and deferred.

Status: accepted for final summary.

Next action: Step L, final feature-analysis summary across XJTU-SY and PHM2012.

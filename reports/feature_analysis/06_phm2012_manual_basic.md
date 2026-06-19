# Step J: PHM2012 manual_basic Three-Task Feature Analysis

## 1. Purpose

Analyze `manual_basic` features on PHM2012 using the official explicit split for three tasks:

1. RUL
2. Health State
3. Early Fault Detection

Step J does not analyze fault type, does not run tsfresh, and does not train models.

## 2. Command

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=phm2012 \
  split=phm2012_official \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=phm2012_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/phm2012
```

## 3. Config

| Item | Value |
|---|---|
| dataset | phm2012 |
| split | phm2012_official |
| feature | manual_basic |
| label | degradation_three_tasks |
| analysis | full_feature_analysis_3tasks |
| run.name | phm2012_3tasks_manual_basic |
| artifact_root | artifacts/feature_analysis |
| feature_source | raw |
| fit_scope | train_only |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/20260619-234112_phm2012_3tasks_manual_basic_44cd0a18/
```

## 5. Files Copied

```text
reports/feature_analysis/phm2012/manual_basic/
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

Selected aggregate curves were copied under:

```text
reports/feature_analysis/phm2012/manual_basic/figures/curves/
```

`selected_curves.txt` records copied and missing requested curves.

## 6. Split Summary

| Split | Bearings | Samples |
|---|---|---:|
| train | `Bearing1_1`, `Bearing1_2`, `Bearing2_1`, `Bearing2_2`, `Bearing3_1`, `Bearing3_2` | 7534 |
| val | `Bearing1_3`, `Bearing2_3` | 4330 |
| test | `Bearing1_4`, `Bearing1_5`, `Bearing1_6`, `Bearing1_7`, `Bearing2_4`, `Bearing2_5`, `Bearing2_6`, `Bearing2_7`, `Bearing3_3` | 13025 |

Split checks in `split_report.json` passed:

- no sample overlap
- no bearing overlap
- non-empty train
- non-empty val
- non-empty test

## 7. Sanity Checks

| Check | Result | Notes |
|---|---:|---|
| `analysis_report.ok` | pass | `true` |
| `analysis_name=full_feature_analysis_3tasks` | pass | Matches Step J config. |
| `feature_source=raw` | pass | Analysis used raw manual features. |
| `fit_scope=train_only` | pass | Scores are fit on the train split only. |
| `num_features=45` | pass | Matches manual_basic expectation. |
| `num_ranked_features=45` | pass | All manual_basic features were ranked. |
| `num_leakage_warnings=1` | pass | One HI/FPT label-source warning found. |
| `plots_enabled` | pass | `true` |
| `feature_ranking exists` | pass | CSV copied to report directory. |
| `feature_cards exists` | pass | CSV copied to report directory. |
| `feature_recommendations exists` | pass | Markdown copied to report directory. |
| `figures exist` | pass | Required figures and selected curves exist. |

`feature_summary.csv` reports 0 missing values, 0 NaN values, 0 Inf values, and no constant split-level feature rows.

## 8. Leakage Summary

| Actual label-source feature | Warning |
|---|---|
| `mag__time__rms` | Feature was used as HI source for FPT-based labels. |

`mag__time__rms` participates in HI/FPT pseudo-label construction. It can be used as a label-source sanity reference, but its HealthState and EarlyFault scores should not be interpreted as independent discovery evidence.

## 9. RUL Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `h__time__mean_abs` | `rul_score=0.544`; Pearson=-0.638, Spearman=-0.659, Kendall=-0.516 | Aggregate curve shows horizontal amplitude rising with degradation on several training bearings. | Bearing-specific trajectories vary. | Strong RUL candidate. |
| 2 | `mag__time__mean` | `rul_score=0.537`; Pearson=-0.569, Spearman=-0.617, Kendall=-0.466 | Magnitude amplitude curve follows the same late-life growth pattern. | Redundant with magnitude mean absolute. | Strong RUL candidate. |
| 2 | `mag__time__mean_abs` | `rul_score=0.537`; Pearson=-0.569, Spearman=-0.617, Kendall=-0.466 | Same visual behavior as `mag__time__mean`. | Redundant with magnitude mean. | Strong RUL candidate. |
| 4 | `h__time__rms` | `rul_score=0.527`; Pearson=-0.602, Spearman=-0.647, Kendall=-0.502 | Horizontal RMS grows with fault progression. | Redundant with `h__time__std`. | Secondary RUL candidate. |
| 5 | `h__time__std` | `rul_score=0.526`; Pearson=-0.603, Spearman=-0.648, Kendall=-0.503 | Similar trend to horizontal RMS. | Redundant with RMS. | Secondary RUL candidate. |

RUL discussion:

- PHM2012 RUL is dominated by amplitude or energy-like time-domain features.
- The strongest independent feature is `h__time__mean_abs`.
- `mag__time__rms` is rank 6, but it is the actual HI/FPT label-source feature and should remain caveated.
- Spectral features are not leading RUL candidates in this run.

## 10. Health State Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `h__time__mean_abs` | `health_score=0.944`; MI=0.486, Fisher=0.789, ANOVA F=1979.4 | Health-state boxplots separate pseudo states strongly. | Pseudo-labels are HI/FPT-derived. | Strong HealthState candidate. |
| 2 | `h__time__std` | `health_score=0.854`; MI=0.474, Fisher=0.682, ANOVA F=1712.1 | Boxplots show clear upward shift across states. | Redundant with horizontal RMS. | Strong HealthState candidate. |
| 3 | `h__time__rms` | `health_score=0.850`; MI=0.469, Fisher=0.682, ANOVA F=1710.7 | Very similar separation to horizontal standard deviation. | Redundant with `h__time__std`. | Strong HealthState candidate. |
| 4 | `mag__time__mean` | `health_score=0.755`; MI=0.495, Fisher=0.531, ANOVA F=1333.4 | Magnitude amplitude supports state separation. | Redundant with magnitude mean absolute. | Stable secondary candidate. |
| 4 | `mag__time__mean_abs` | `health_score=0.755`; MI=0.495, Fisher=0.531, ANOVA F=1333.4 | Same pattern as magnitude mean. | Redundant with magnitude mean. | Stable secondary candidate. |

HealthState discussion:

- PHM2012 HealthState findings are strongly horizontal-channel and amplitude-driven.
- This agrees with XJTU-SY Step F more than with Step I's train-condition-specific vertical-channel emphasis.
- `mag__spectral__entropy` appears as a secondary HealthState feature, but the main recommendation remains amplitude features.

## 11. Early Fault Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `h__time__mean_abs` | `early_fault_score=0.994`; AUC=0.836, Cohen's d=1.000, mean shift=0.220 | EarlyFault plot shows post-FPT amplitude increase. | Pseudo-label boundary comes from HI/FPT. | Strong EarlyFault candidate. |
| 2 | `mag__time__mean` | `early_fault_score=0.983`; AUC=0.824, Cohen's d=1.012, mean shift=0.317 | Magnitude amplitude separates pre/post FPT. | Redundant with magnitude mean absolute. | Strong EarlyFault candidate. |
| 2 | `mag__time__mean_abs` | `early_fault_score=0.983`; AUC=0.824, Cohen's d=1.012, mean shift=0.317 | Same visual behavior as magnitude mean. | Redundant with magnitude mean. | Strong EarlyFault candidate. |
| 4 | `h__time__std` | `early_fault_score=0.908`; AUC=0.817, Cohen's d=0.881, mean shift=0.296 | Horizontal dispersion increases after FPT. | Redundant with horizontal RMS. | Strong secondary candidate. |
| 5 | `h__time__rms` | `early_fault_score=0.906`; AUC=0.817, Cohen's d=0.880, mean shift=0.296 | Similar pre/post separation to standard deviation. | Redundant with `h__time__std`. | Strong secondary candidate. |

EarlyFault discussion:

- PHM2012 EarlyFault is also dominated by amplitude features.
- Spectral features such as `v__spectral__rms_frequency` and `v__spectral__peak_frequency` appear around ranks 11-12, so they are secondary rather than mainline features.
- The label-source caveat applies to `mag__time__rms`, but not to `h__time__mean_abs`, `h__time__std`, or `h__time__rms` in this run.

## 12. Distribution Check

| Feature | Train Mean | Val Mean | Test Mean | Observation |
|---|---:|---:|---:|---|
| `h__time__mean_abs` | 0.400140 | 0.427395 | 0.383719 | Means are close, but val/test variance is larger. |
| `h__time__std` | 0.527971 | 0.577499 | 0.493517 | Similar means with larger val/test spread. |
| `h__time__rms` | 0.528463 | 0.578020 | 0.494124 | Similar to horizontal standard deviation. |
| `mag__time__mean` | 0.631520 | 0.683824 | 0.714782 | Test mean is slightly higher and more variable. |
| `mag__time__mean_abs` | 0.631520 | 0.683824 | 0.714782 | Same as magnitude mean. |
| `mag__time__rms` | 0.763901 | 0.931635 | 0.833521 | Label-source feature; val/test variance is much larger. |
| `v__time__mean_abs` | 0.402712 | 0.423311 | 0.519667 | Vertical amplitude is more variable in test. |
| `v__time__std` | 0.527353 | 0.671319 | 0.664825 | Val/test means and spreads are higher. |

Distribution discussion:

- Unlike XJTU Step I, the top PHM2012 horizontal amplitude features have fairly close train/val/test means.
- Val/test sets show larger maxima and standard deviations, so downstream models still need split-aware validation.
- `mag__time__rms` remains the most important caveated feature because it is the HI/FPT source.

## 13. Cross-Dataset Comparison Notes

| Topic | PHM2012 Observation | XJTU-SY Observation | Interpretation |
|---|---|---|---|
| RUL top feature family | Horizontal and magnitude amplitude features dominate. | Step F and Step H also favor magnitude/horizontal amplitude; Step I keeps RUL amplitude features stable. | RUL amplitude features are consistent across datasets. |
| HealthState top feature family | `h__time__mean_abs`, `h__time__std`, and `h__time__rms` dominate. | Step F favors the same horizontal amplitude family; Step I downgrades some horizontal features because ranking is train-condition-specific. | PHM2012 supports Step F's horizontal HealthState finding. |
| EarlyFault top feature family | Horizontal and magnitude amplitude dominate. | Step F favors horizontal amplitude; Step H shows EarlyFault is condition-sensitive; Step I highlights Condition 1 spectral effects. | PHM2012 supports amplitude as the mainline EarlyFault family, with spectral features secondary. |
| Label-source caveat | `mag__time__rms` is the actual label-source feature. | Same actual label-source feature in XJTU-SY Step F/H/I. | Keep `mag__time__rms` as reference, not independent evidence. |
| Distribution shift | Top horizontal means are close across split, but val/test variance is larger. | XJTU Step I has clearer condition-driven mean shift. | PHM2012 official split is less obviously mean-shifted for the top horizontal features, but variance remains important. |

Detailed cross-dataset comparison is stored in:

```text
reports/feature_analysis/summary/phm2012_vs_xjtu_manual_basic_check.csv
```

## 14. Figures Reviewed

Reviewed required figures:

- `figures/rul_top_features.png`
- `figures/degradation_score_heatmap.png`
- `figures/health_state_boxplots.png`
- `figures/early_fault_effects.png`
- `figures/feature_recommendation_matrix.png`
- `figures/feature_score_heatmap.png`

Copied selected aggregate curves:

- `figures/curves/h__time__mean_abs.png`
- `figures/curves/h__time__std.png`
- `figures/curves/h__time__rms.png`
- `figures/curves/mag__time__mean.png`
- `figures/curves/mag__time__mean_abs.png`
- `figures/curves/mag__time__rms.png`
- `figures/curves/v__time__mean_abs.png`
- `figures/curves/v__time__std.png`

All requested selected aggregate curves were generated and copied. All copied PNG files were checked as nonblank.

## 15. Issues / Warnings

- PHM2012 split / label caveat: HealthState and EarlyFault labels are pseudo-labels derived through HI/FPT logic.
- Data quality: no missing, NaN, Inf, or constant split-level feature summaries were observed.
- Leakage: `mag__time__rms` is the actual HI/FPT label-source feature.
- Plot quality: required figures and selected curves are present and readable.
- Other: `analysis_report.enabled_sections` includes summary, RUL correlation, degradation scores, HealthState, and EarlyFault. Fault type is not part of this three-task mainline.

## 16. Decision

- [ ] Pass
- [x] Needs review
- [ ] Needs rerun
- [ ] Blocked

Next action: review Step J PHM2012 `manual_basic` findings. After acceptance, proceed to Step K, PHM2012 `manual_tsfresh_basic` comparison attempt.

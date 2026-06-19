# Step F: XJTU-SY Main Feature Analysis with manual_basic

## 1. Purpose

Analyze `manual_basic` features on XJTU-SY using the main all-condition bearing-index split for three tasks:

1. RUL
2. Health State
3. Early Fault Detection

Step F does not run PHM2012, does not train models, and does not copy generated feature/label parquet artifacts into reports.

## 2. Command

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_bearing_index_split \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_all_conditions_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

## 3. Config

| Item | Value |
|---|---|
| dataset | xjtu_sy |
| split | xjtu_bearing_index_split |
| feature | manual_basic |
| label | degradation_three_tasks |
| analysis | full_feature_analysis_3tasks |
| run.name | xjtu_all_conditions_3tasks_manual_basic |
| artifact_root | artifacts/feature_analysis |
| feature_source | raw |
| fit_scope | train_only |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/20260619-215058_xjtu_all_conditions_3tasks_manual_basic_a41a36c4/
```

## 5. Files Copied

```text
reports/feature_analysis/xjtu_sy/all_conditions_bearing_index_manual_basic/
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

Selected curves were copied for the de-duplicated top features across RUL, Health State, and Early Fault:

```text
figures/curves/mag__time__rms.png
figures/curves/mag__time__mean.png
figures/curves/mag__time__mean_abs.png
figures/curves/h__time__mean_abs.png
figures/curves/h__time__rms.png
figures/curves/h__time__std.png
```

## 6. Sanity Checks

| Check | Result | Notes |
|---|---:|---|
| `analysis_report.ok` | pass | `true` |
| `analysis_name=full_feature_analysis_3tasks` | pass | Matches Step F config. |
| `feature_source=raw` | pass | Analysis used raw manual features. |
| `fit_scope=train_only` | pass | Ranking and analysis are fit on train scope. |
| `num_features=45` | pass | Matches Step E feature extraction sanity. |
| `num_ranked_features=45` | pass | All manual_basic features were ranked. |
| `feature_ranking exists` | pass | CSV copied to report directory. |
| `feature_cards exists` | pass | CSV copied to report directory. |
| `feature_recommendations exists` | pass | Markdown copied to report directory. |
| `leakage_report checked` | pass | One HI/FPT label-source warning found. |
| `figures exist` | pass | Required figures and selected curves exist and are nonblank. |

`feature_summary.csv` reports 0 missing values, 0 NaN values, 0 Inf values, and 0 constant feature rows across the split-level summaries.

## 7. Leakage Warnings

`leakage_report.json` contains one warning:

```text
mag__time__rms: Feature was used as HI source for FPT-based labels.
```

Caveat: `mag__time__rms` participates in HI/FPT pseudo-label construction. Its strong RUL, Health State, or Early Fault performance should not be interpreted as a completely independent discovery signal. It can still be useful as a label-source sanity reference, but it should be separated from candidate features when making claims about independent predictive value.

The label config also lists `h__time__rms` and `v__time__rms` as HI source candidates, but this run selected `mag__time__rms` as the actual label-source feature.

## 8. RUL Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `mag__time__rms` | `rul_score=0.625`; Pearson=-0.786, Spearman=-0.764, Kendall=-0.624 | Curve rises sharply near late-life regions for several bearings. | Label-source feature for HI/FPT labels. | Useful as reference, not independent evidence. |
| 2 | `mag__time__mean` | `rul_score=0.625`; Pearson=-0.792, Spearman=-0.768, Kendall=-0.628 | Similar late-life upward trend to magnitude RMS. | Not flagged as label-source, but highly related to magnitude energy. | Candidate RUL trend feature. |
| 3 | `mag__time__mean_abs` | `rul_score=0.625`; Pearson=-0.792, Spearman=-0.768, Kendall=-0.628 | Similar monotone degradation-like shape across selected bearings. | Not flagged as label-source, but redundant with magnitude mean in this run. | Candidate RUL trend feature. |

Notes:

- RUL top features are mostly magnitude or energy-like time-domain features.
- The strongest numerical RUL feature is also the HI/FPT label-source feature, so it needs explicit caveat in any downstream interpretation.
- The selected curves show clear late-stage increases, especially for `mag__time__rms`, `mag__time__mean`, and `mag__time__mean_abs`.

## 9. Health State Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `h__time__mean_abs` | `health_score=0.996`; mutual information=0.623, Fisher=1.864, ANOVA F=4366.3 | Health-state boxplots separate severe state clearly; curve rises with degradation. | Horizontal channel amplitude can be condition-sensitive. | Strong Health State candidate. |
| 2 | `h__time__rms` | `health_score=0.977`; mutual information=0.629, Fisher=1.792, ANOVA F=4198.8 | Boxplot and curves show clear state separation. | Not flagged as actual label-source in this run, but it is listed as an HI source candidate in config. | Strong candidate with caveat. |
| 3 | `h__time__std` | `health_score=0.975`; mutual information=0.626, Fisher=1.793, ANOVA F=4199.5 | Very similar separation pattern to horizontal RMS. | Likely redundant with RMS for many bearings. | Strong but redundant candidate. |

Notes:

- Health-state separation is dominated by horizontal time-domain amplitude features.
- Boxplots show the severe state shifting upward strongly, while early states still overlap more.
- `h__time__rms` should be interpreted carefully because it is configured as an HI source candidate, even though the actual leakage warning selected `mag__time__rms`.

## 10. Early Fault Feature Findings

| Rank | Feature | Score Evidence | Plot Evidence | Caveat | Decision |
|---:|---|---|---|---|---|
| 1 | `h__time__mean_abs` | `early_fault_score=0.996`; AUC=0.949, Cohen's d=1.514, mean shift=0.910 | Early-fault boxplot shows post-FPT samples shifted upward. | Can react to load/condition amplitude changes. | Strong Early Fault candidate. |
| 2 | `h__time__std` | `early_fault_score=0.991`; AUC=0.950, Cohen's d=1.495, mean shift=1.166 | Curves and early-fault effect plot show clear before/after separation. | Redundant with horizontal RMS. | Strong Early Fault candidate. |
| 3 | `h__time__rms` | `early_fault_score=0.991`; AUC=0.950, Cohen's d=1.494, mean shift=1.165 | Similar before/after separation to standard deviation. | Configured as an HI source candidate, although not the actual warning feature. | Strong candidate with caveat. |

Notes:

- Early Fault top features have high AUC and large effect sizes.
- The results are physically plausible because amplitude and dispersion increase after degradation onset.
- These features should be cross-checked in later condition-wise or cross-condition steps because energy features can be sensitive to operating condition and bearing-specific failure speed.

## 11. Figures Reviewed

Reviewed required figures:

- `figures/rul_top_features.png`
- `figures/degradation_score_heatmap.png`
- `figures/health_state_boxplots.png`
- `figures/early_fault_effects.png`
- `figures/feature_recommendation_matrix.png`
- `figures/feature_score_heatmap.png`

Reviewed selected curves:

- `figures/curves/mag__time__rms.png`
- `figures/curves/mag__time__mean.png`
- `figures/curves/mag__time__mean_abs.png`
- `figures/curves/h__time__mean_abs.png`
- `figures/curves/h__time__rms.png`
- `figures/curves/h__time__std.png`

Visual review summary:

- RUL top features show degradation-like late-life increases.
- Health-state boxplots show stronger separation for severe state than for early states.
- EarlyFault effect plots show post-FPT distribution shifts for the top horizontal amplitude features.
- Figure files are present and nonblank.

## 12. Issues / Warnings

- Data quality: no missing, NaN, Inf, or constant split-level feature summaries were observed.
- Distribution shift: energy-like features show strong bearing-specific curve shapes; later condition-wise and cross-condition checks are still needed.
- Leakage: `mag__time__rms` was used as HI/FPT label source and must be caveated.
- Plot quality: plots are readable and nonblank. `feature_recommendations.md` still contains a generated Fault Type section even though Step F does not analyze fault type; this section is ignored for the three-task mainline.
- Other: `analysis_report.enabled_sections` includes summary, RUL correlation, degradation scores, health state, and early fault; it does not include fault type.

## 13. Decision

- [ ] Pass
- [x] Needs review
- [ ] Needs rerun
- [ ] Blocked

Next action: review the Step F report and figures. After acceptance, proceed to Step G, XJTU-SY main split with `manual_tsfresh_basic`.

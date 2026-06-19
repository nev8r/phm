# Step I: XJTU-SY Cross-Condition manual_basic Robustness Analysis

## 1. Purpose

Evaluate whether the XJTU-SY `manual_basic` findings remain usable under a cross-condition split.

This step uses:

- train: `35Hz12kN`
- val: `37.5Hz11kN`
- test: `40Hz10kN`

Important caveat: feature ranking is still train-only. The ranking is computed from the train condition, `35Hz12kN`, only. Validation and test splits are used for distribution and visualization checks across unseen operating conditions, not for fitting the ranking scores.

Step I does not run tsfresh, PHM2012, model training, checkpointing, or prediction export.

## 2. Command

```bash
uv run bp --config-name smoke \
  mode=analyze_features \
  dataset=xjtu_sy \
  split=xjtu_cross_condition \
  feature=manual_basic \
  label=degradation_three_tasks \
  analysis=full_feature_analysis_3tasks \
  run.name=xjtu_cross_condition_3tasks_manual_basic \
  project.artifact_root=artifacts/feature_analysis \
  dataset.root=data/loader_roots/xjtu
```

## 3. Config

| Item | Value |
|---|---|
| dataset | xjtu_sy |
| split | xjtu_cross_condition |
| feature | manual_basic |
| label | degradation_three_tasks |
| analysis | full_feature_analysis_3tasks |
| run.name | xjtu_cross_condition_3tasks_manual_basic |
| artifact_root | artifacts/feature_analysis |
| feature_source | raw |
| fit_scope | train_only |

## 4. Run Directory

```text
artifacts/feature_analysis/runs/20260619-232459_xjtu_cross_condition_3tasks_manual_basic_49469df1/
```

## 5. Files Copied

```text
reports/feature_analysis/xjtu_sy/cross_condition_manual_basic/
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
reports/feature_analysis/xjtu_sy/cross_condition_manual_basic/figures/curves/
```

`selected_curves.txt` records copied and missing requested curves.

## 6. Split Summary

| Split | Condition | Bearings | Samples |
|---|---|---|---:|
| train | `35Hz12kN` | `Bearing1_1` to `Bearing1_5` | 616 |
| val | `37.5Hz11kN` | `Bearing2_1` to `Bearing2_5` | 1566 |
| test | `40Hz10kN` | `Bearing3_1` to `Bearing3_5` | 7034 |

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
| `analysis_name=full_feature_analysis_3tasks` | pass | Matches Step I config. |
| `feature_source=raw` | pass | Analysis used raw manual features. |
| `fit_scope=train_only` | pass | Scores are fit on the train condition only. |
| `num_features=45` | pass | Matches manual_basic expectation. |
| `num_ranked_features=45` | pass | All manual_basic features were ranked. |
| `num_leakage_warnings=1` | pass | One HI/FPT label-source warning found. |
| `feature_ranking exists` | pass | CSV copied to report directory. |
| `feature_cards exists` | pass | CSV copied to report directory. |
| `feature_recommendations exists` | pass | Markdown copied to report directory. |
| `figures exist` | pass | Required figures and selected available curves exist. |

`feature_summary.csv` reports 0 missing values, 0 NaN values, 0 Inf values, and no constant split-level feature rows.

## 8. Leakage Summary

| Actual label-source feature | Warning |
|---|---|
| `mag__time__rms` | Feature was used as HI source for FPT-based labels. |

`mag__time__rms` remains useful as a label-source reference, but it should not be overclaimed as independent evidence for Health State or Early Fault detection.

## 9. Cross-Condition Distribution Check

The table below focuses on Step F/H core features. The main pattern is that many amplitude features have lower test-condition means than train-condition means, which indicates operating-condition sensitivity.

| Feature | Train Mean | Val Mean | Test Mean | Shift Note | Decision |
|---|---:|---:|---:|---|---|
| `mag__time__rms` | 2.221831 | 2.519407 | 1.340987 | test lower than train | label-source reference |
| `mag__time__mean` | 1.922017 | 2.207957 | 1.168386 | test lower than train | stable RUL candidate |
| `mag__time__mean_abs` | 1.922017 | 2.207957 | 1.168386 | test lower than train | stable RUL candidate |
| `h__time__mean_abs` | 1.121159 | 1.296775 | 0.708641 | test lower than train | condition-sensitive for Health/EarlyFault |
| `h__time__rms` | 1.445736 | 1.642115 | 0.897779 | test lower than train | condition-sensitive for Health/EarlyFault |
| `h__time__std` | 1.444930 | 1.641659 | 0.897419 | test lower than train | condition-sensitive for Health/EarlyFault |
| `mag__time__std` | 1.111336 | 1.210262 | 0.656163 | test lower than train | stable secondary candidate |
| `v__time__mean_abs` | 1.323545 | 1.499203 | 0.776414 | test lower than train | strong train-condition candidate |
| `v__time__std` | 1.660538 | 1.877701 | 0.987162 | test lower than train | strong train-condition candidate |
| `v__spectral__entropy` | 0.673236 | 0.632018 | 0.745154 | low mean shift | condition-specific EarlyFault signal |

Detailed comparison is stored in:

```text
reports/feature_analysis/summary/xjtu_cross_condition_feature_check.csv
```

## 10. RUL Robustness

| Feature | Step F Rank | Step H Top10 Count | Step I Rank | Interpretation |
|---|---:|---:|---:|---|
| `mag__time__mean` | 2 | 3 | 4 | Survives Step I as a stable RUL candidate. |
| `mag__time__mean_abs` | 2 | 3 | 4 | Same behavior as magnitude mean; redundant but stable. |
| `mag__time__rms` | 1 | 3 | 6 | Stable ranking, but label-source caveat applies. |
| `h__time__mean_abs` | 4 | 3 | 8 | Still top-10 for RUL, but less dominant than in Step F. |
| `h__time__rms` | 5 | 3 | 10 | Barely stays top-10; keep as secondary RUL candidate. |
| `h__time__std` | 6 | 3 | 9 | Still top-10; redundant with horizontal RMS. |
| `mag__time__std` | 8 | 3 | 7 | Stable secondary RUL candidate. |
| `v__time__mean_abs` | 7 | 3 | 1 | Becomes the strongest train-condition RUL feature. |
| `v__time__std` | 10 | 3 | 3 | Strong in Step I and stable across Step H top-10 counts. |

RUL conclusion:

- Step F/H RUL amplitude features mostly survive the cross-condition split.
- Vertical-channel amplitude features rise to the top because the train-only ranking is fit on Condition 1.
- `mag__time__rms` remains a label-source reference rather than independent evidence.

## 11. Health State Robustness

| Feature | Step F Rank | Step H Top10 Count | Step I Rank | Interpretation |
|---|---:|---:|---:|---|
| `h__time__mean_abs` | 1 | 3 | 14 | Downgraded under train-only cross-condition ranking. |
| `h__time__rms` | 2 | 2 | 13 | Downgraded; no longer a robust cross-condition top feature. |
| `h__time__std` | 3 | 2 | 10 | Borderline top-10; keep as secondary. |
| `mag__time__mean` | 4 | 3 | 5 | Stable HealthState candidate. |
| `mag__time__mean_abs` | 4 | 3 | 5 | Stable, redundant with magnitude mean. |
| `mag__time__rms` | 6 | 3 | 4 | Stable but label-source caveat applies. |
| `mag__time__std` | 7 | 3 | 7 | Stable secondary HealthState candidate. |
| `v__time__mean_abs` | 9 | 2 | 1 | Strong in Step I; likely Condition 1 sensitive. |
| `v__time__std` | 11 | 2 | 3 | Strong in Step I; should be treated as condition-sensitive. |

Health State conclusion:

- Step F's horizontal-channel HealthState story does not fully survive Step I.
- Magnitude features are more stable across Step F/H/I than pure horizontal features.
- Vertical amplitude features dominate the train-condition ranking, so they should be validated carefully before being treated as global cross-condition features.

## 12. Early Fault Robustness

| Feature | Step F Rank | Step H Top10 Count | Step I Rank | Interpretation |
|---|---:|---:|---:|---|
| `h__time__mean_abs` | 1 | 1 | 16 | Downgraded; strong in C3 but not robust in Step I. |
| `h__time__std` | 2 | 1 | 14 | Downgraded; condition-specific horizontal signal. |
| `h__time__rms` | 3 | 1 | 13 | Downgraded; condition-specific horizontal signal. |
| `mag__time__mean` | 4 | 3 | 9 | Survives as a stable EarlyFault candidate. |
| `mag__time__mean_abs` | 4 | 3 | 9 | Stable, redundant with magnitude mean. |
| `mag__time__rms` | 6 | 3 | 8 | Stable but label-source caveat applies. |
| `mag__time__std` | 7 | 2 | 7 | Stable secondary EarlyFault candidate. |
| `v__time__mean_abs` | 9 | 2 | 5 | Survives Step I but remains condition-sensitive. |
| `v__time__std` | 10 | 3 | 3 | Strong Step I EarlyFault candidate. |
| `v__spectral__entropy` | 29 | 1 | 1 | Condition 1 spectral signal; do not generalize as global. |
| `mag__spectral__entropy` | 16 | 1 | 2 | Condition-specific spectral signal. |

Early Fault conclusion:

- Step F's horizontal amplitude EarlyFault features should be downgraded from global candidates to condition-sensitive candidates.
- Magnitude amplitude features remain the safer cross-step baseline.
- Spectral entropy dominates Step I because ranking is fit on Condition 1; this matches Step H's observation that C1 has condition-specific spectral EarlyFault behavior.

## 13. Figures Reviewed

Reviewed required figures:

- `figures/rul_top_features.png`
- `figures/degradation_score_heatmap.png`
- `figures/health_state_boxplots.png`
- `figures/early_fault_effects.png`
- `figures/feature_recommendation_matrix.png`
- `figures/feature_score_heatmap.png`

Copied selected aggregate curves:

- `figures/curves/mag__time__rms.png`
- `figures/curves/mag__time__mean.png`
- `figures/curves/mag__time__mean_abs.png`
- `figures/curves/mag__time__std.png`
- `figures/curves/v__time__mean_abs.png`
- `figures/curves/v__time__std.png`
- `figures/curves/v__spectral__entropy.png`
- `figures/curves/v__time__ptp.png`

Requested but not generated by the plotting utility:

- `h__time__mean_abs`
- `h__time__rms`
- `h__time__std`
- `mag__time__ptp`

All copied PNG files were checked as nonblank.

## 14. Issues / Warnings

- Train-only ranking caveat: Step I ranks features using only `35Hz12kN`, so top features should not be described as a true all-condition ranking.
- Distribution shift: many amplitude features have lower test-condition means than train-condition means; downstream models need train-only scaling and split-aware validation.
- Leakage: `mag__time__rms` is the actual HI/FPT label-source feature.
- Plot coverage: the analysis plotting utility did not generate some requested horizontal aggregate curves; missing entries are recorded in `selected_curves.txt`.
- Interpretation: Step I supports `manual_basic` as the current XJTU-SY mainline feature set, but it downgrades horizontal HealthState/EarlyFault claims to condition-sensitive claims.

## 15. Decision

- [ ] Pass
- [x] Needs review
- [ ] Needs rerun
- [ ] Blocked

Next action: review Step I cross-condition robustness findings. After acceptance, proceed to Step J, PHM2012 official `manual_basic` three-task analysis.

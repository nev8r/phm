# Step S: Full Feature Baseline Training Batch

## 1. Purpose

Train full_manual_basic_no_reference and full_manual_basic MLP baselines and compare them with compact subsets.

This step is real training, not a dry run. It completes the main split / official split feature-subset grid across compact non-reference, compact reference, full non-reference, and full reference settings.

## 2. Experiments

| ID | Dataset | Task | Feature Subset | Feature Count | Run Dir | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| S1 | XJTU-SY | RUL | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-165326_xjtu_main_rul_mlp_full_manual_basic_no_reference_c0d6f3ac` | completed |
| S2 | XJTU-SY | RUL | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-165829_xjtu_main_rul_mlp_full_manual_basic_d0c53668` | completed |
| S3 | XJTU-SY | HealthState | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-170311_xjtu_main_health_mlp_full_manual_basic_no_reference_9ebfd46d` | completed |
| S4 | XJTU-SY | HealthState | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-170756_xjtu_main_health_mlp_full_manual_basic_2ed9942f` | completed |
| S5 | XJTU-SY | EarlyFault | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-171242_xjtu_main_early_mlp_full_manual_basic_no_reference_e0ce46bd` | completed |
| S6 | XJTU-SY | EarlyFault | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-171716_xjtu_main_early_mlp_full_manual_basic_8c2c8626` | completed |
| S7 | PHM2012 | RUL | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-172133_phm_official_rul_mlp_full_manual_basic_no_reference_3ad07f96` | completed |
| S8 | PHM2012 | RUL | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-172607_phm_official_rul_mlp_full_manual_basic_d8241feb` | completed |
| S9 | PHM2012 | HealthState | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-173102_phm_official_health_mlp_full_manual_basic_no_reference_490e9ded` | completed |
| S10 | PHM2012 | HealthState | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-173551_phm_official_health_mlp_full_manual_basic_1e660f03` | completed |
| S11 | PHM2012 | EarlyFault | `full_manual_basic_no_reference` | 44 | `artifacts/baselines/runs/20260621-174100_phm_official_early_mlp_full_manual_basic_no_reference_d53b5bba` | completed |
| S12 | PHM2012 | EarlyFault | `full_manual_basic` | 45 | `artifacts/baselines/runs/20260621-174612_phm_official_early_mlp_full_manual_basic_aead21f0` | completed |

## 3. Training Completion Checks

| ID | max_epochs | last_epoch | best_epoch | best_metric | checkpoints | val/test predictions | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| S1 | 50 | 50 | 13 | 0.046904 | yes | yes | pass |
| S2 | 50 | 50 | 7 | 0.049000 | yes | yes | pass |
| S3 | 50 | 50 | 1 | 1.380101 | yes | yes | pass |
| S4 | 50 | 50 | 1 | 1.339337 | yes | yes | pass |
| S5 | 50 | 50 | 1 | 0.977301 | yes | yes | pass |
| S6 | 50 | 50 | 1 | 1.000753 | yes | yes | pass |
| S7 | 50 | 50 | 31 | 0.152141 | yes | yes | pass |
| S8 | 50 | 50 | 1 | 0.119870 | yes | yes | pass |
| S9 | 50 | 50 | 1 | 1.796848 | yes | yes | pass |
| S10 | 50 | 50 | 1 | 1.619039 | yes | yes | pass |
| S11 | 50 | 50 | 1 | 0.513370 | yes | yes | pass |
| S12 | 50 | 50 | 2 | 0.500975 | yes | yes | pass |

Every Step S run has `history.json` length 50, final history epoch 50, `trainer_state.epoch` 50, expected feature count, and required raw outputs. The no-reference runs exclude `mag__time__rms`; the full reference runs include it.

## 4. Metrics Summary

| Experiment | Primary Metric | Val | Test | Notes |
| --- | --- | ---: | ---: | --- |
| S1 XJTU-SY RUL full_manual_basic_no_reference | RMSE | 0.339505 | 0.421645 | lower is better |
| S2 XJTU-SY RUL full_manual_basic | RMSE | 0.462780 | 0.440693 | lower is better |
| S3 XJTU-SY HealthState full_manual_basic_no_reference | WeightedF1 | 0.610433 | 0.227696 | higher is better |
| S4 XJTU-SY HealthState full_manual_basic | WeightedF1 | 0.597020 | 0.213177 | higher is better |
| S5 XJTU-SY EarlyFault full_manual_basic_no_reference | WeightedF1 | 0.678678 | 0.576118 | higher is better |
| S6 XJTU-SY EarlyFault full_manual_basic | WeightedF1 | 0.675783 | 0.577494 | higher is better |
| S7 PHM2012 RUL full_manual_basic_no_reference | RMSE | 0.474238 | 0.408619 | lower is better |
| S8 PHM2012 RUL full_manual_basic | RMSE | 0.421059 | 0.334945 | lower is better |
| S9 PHM2012 HealthState full_manual_basic_no_reference | WeightedF1 | 0.290378 | 0.339013 | higher is better |
| S10 PHM2012 HealthState full_manual_basic | WeightedF1 | 0.270328 | 0.304306 | higher is better |
| S11 PHM2012 EarlyFault full_manual_basic_no_reference | WeightedF1 | 0.526412 | 0.472495 | higher is better |
| S12 PHM2012 EarlyFault full_manual_basic | WeightedF1 | 0.526063 | 0.487209 | higher is better |

## 5. Full vs Compact Comparison

| Dataset | Task | Compact Non-ref | Full Non-ref | Compact Ref | Full Ref | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| XJTU-SY | rul_tabular | 0.428591 | 0.421645 | 0.468908 | 0.440693 | Full features improved both non-reference and reference comparisons. |
| XJTU-SY | health_state_tabular | 0.371101 | 0.227696 | 0.368269 | 0.213177 | Full features did not improve test primary metrics in either comparison. |
| XJTU-SY | early_fault_tabular | 0.841682 | 0.576118 | 0.841682 | 0.577494 | Full features did not improve test primary metrics in either comparison. |
| PHM2012 | rul_tabular | 0.392475 | 0.408619 | 0.360850 | 0.334945 | Full reference improved, while full non-reference did not improve over compact non-reference. |
| PHM2012 | health_state_tabular | 0.406725 | 0.339013 | 0.417337 | 0.304306 | Full features did not improve test primary metrics in either comparison. |
| PHM2012 | early_fault_tabular | 0.664556 | 0.472495 | 0.672350 | 0.487209 | Full features did not improve test primary metrics in either comparison. |

For RUL, the primary metric is RMSE and lower is better. The full-feature effect is `compact_RMSE - full_RMSE`, so a positive value means the full feature set improved RMSE. For HealthState and EarlyFault, the primary metric is WeightedF1 and higher is better. The full-feature effect is `full_WeightedF1 - compact_WeightedF1`, so a positive value means the full feature set improved WeightedF1.

## 6. Findings

### RUL

Full features are mixed. XJTU-SY RUL improves slightly in the non-reference comparison but worsens in the reference comparison. PHM2012 RUL improves clearly for the full reference comparison but worsens for the no-reference comparison.

### HealthState

Full features do not improve HealthState on either dataset in these test metrics. XJTU-SY HealthState and PHM2012 HealthState both show lower full-feature WeightedF1 than their compact counterparts.

### EarlyFault

Full features degrade XJTU-SY EarlyFault relative to compact subsets, but PHM2012 EarlyFault shows improvement for the full no-reference and full reference comparisons.

## 7. Caveats

- `full_manual_basic` includes `mag__time__rms`.
- `full_manual_basic_no_reference` excludes `mag__time__rms`.
- HealthState and EarlyFault are pseudo-label tasks derived from the HI/FPT labeling pipeline.
- Any HealthState/EarlyFault gain from `full_manual_basic` may reflect the label-source feature rather than independent feature evidence.
- This is still MLP only and not a tuned model.

## 8. Decision

- [x] Pass to review.
- [ ] Needs rerun.
- [ ] Blocked.

Next action: Step T should summarize Step Q/R/S and give the final compact vs full vs reference conclusion for the main split / official split baseline cycle.

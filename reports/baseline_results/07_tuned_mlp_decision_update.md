# Step X: Tuned MLP Decision Update

## 1. Purpose

Merge Step W tuned MLP pilot results into the final baseline decision table.

No new training, evaluation, checkpoint creation, prediction export, feature extraction, or raw artifact copy is run in this step. Step X only consolidates curated Step V and Step W result files.

## 2. Inputs

- `FINAL_BASELINE_REPORT.md`
- `baseline_final_decisions.csv`
- `06_tuned_mlp_pilot.md`
- `tuned_mlp_pilot_metrics.csv`
- `tuned_vs_default_mlp_comparison.csv`

## 3. Tuned Setting

| Field | Default | Tuned |
| --- | ---: | ---: |
| hidden_size | 64 | 128 |
| batch_size | 16 | 64 |
| lr | 0.001 | 0.0005 |
| weight_decay | 0.0 | 0.0001 |
| max_epochs | 50 | 50 |

## 4. Default vs Tuned Summary

| Dataset | Task | Metric | Default Test | Tuned Test | Tuned Effect | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| XJTU-SY | RUL | RMSE | 0.421645 | 0.443106 | -0.021461 | keep default MLP |
| XJTU-SY | HealthState | WeightedF1 | 0.371101 | 0.359146 | -0.011955 | keep default MLP |
| XJTU-SY | EarlyFault | WeightedF1 | 0.841682 | 0.837047 | -0.004635 | keep default MLP |
| PHM2012 | RUL | RMSE | 0.392475 | 0.337661 | 0.054814 | tuned MLP improves test metric; keep tuned as PHM2012 RUL candidate |
| PHM2012 | HealthState | WeightedF1 | 0.406725 | 0.441598 | 0.034874 | tuned MLP improves test metric; keep tuned as PHM2012 HealthState candidate |
| PHM2012 | EarlyFault | WeightedF1 | 0.664556 | 0.679212 | 0.014657 | tuned MLP improves test metric; keep tuned as PHM2012 EarlyFault candidate |

For RUL, RMSE is lower-is-better and positive tuned effect means tuned reduced RMSE. For HealthState and EarlyFault, WeightedF1 is higher-is-better and positive tuned effect means tuned improved WeightedF1.

## 5. Dataset-Level Interpretation

### XJTU-SY

Tuned MLP worsened all three test primary metrics. Keep default MLP for the first baseline report on XJTU-SY. The feature subset decisions do not change: RUL keeps `full_manual_basic_no_reference`, and HealthState/EarlyFault keep `compact_non_label_source`.

### PHM2012

Tuned MLP improved all three test primary metrics, but validation metrics were worse in the pilot. Keep tuned MLP as a PHM2012 candidate, not as a universally final replacement without repeat runs or a small sweep. The feature subset decisions do not change: all PHM2012 independent recommendations remain `compact_non_label_source`.

## 6. Final Model Decision

| Dataset | Task | Recommended Model | Recommended Feature Subset | Caveat |
| --- | --- | --- | --- | --- |
| XJTU-SY | RUL | default MLP | `full_manual_basic_no_reference` | Tuned MLP worsened test RMSE; keep default MLP with full_manual_basic_no_reference. |
| XJTU-SY | HealthState | default MLP | `compact_non_label_source` | Tuned MLP worsened test WeightedF1; HealthState remains a pseudo-label task. |
| XJTU-SY | EarlyFault | default MLP | `compact_non_label_source` | Tuned MLP worsened test WeightedF1; EarlyFault is already condition-sensitive. |
| PHM2012 | RUL | tuned MLP candidate | `compact_non_label_source` | Validation RMSE worsened, so do not declare tuned universally superior without repeat or sweep. |
| PHM2012 | HealthState | tuned MLP candidate | `compact_non_label_source` | Validation WeightedF1 worsened and HealthState is a pseudo-label task. |
| PHM2012 | EarlyFault | tuned MLP candidate | `compact_non_label_source` | Validation WeightedF1 worsened and EarlyFault is a pseudo-label task. |

The machine-readable decision update is `baseline_final_decisions_with_tuned.csv`.

## 7. Caveats

- Tuned MLP is one pilot setting, not a hyperparameter search.
- Validation/test consistency is mixed, especially for PHM2012 where test improves while validation worsens.
- HealthState and EarlyFault are pseudo-label tasks.
- All tuned runs use independent non-reference feature subsets.
- No tuned cross-condition run is included yet.

## 8. Decision

- [x] Pass to review.
- [x] Keep default MLP for XJTU-SY in the current baseline report.
- [x] Keep tuned MLP as a PHM2012 candidate, pending repeat or sweep validation.
- [x] Keep all Step V independent feature subset decisions unchanged.
- [ ] Needs fix.
- [ ] Blocked.

Next action: either close the MLP baseline phase for reporting, or open Step Y for non-MLP tabular baseline planning.

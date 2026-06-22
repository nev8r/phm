# Step AB: GRU Sequence 200ep Batch

## 1. Purpose

Run six 200-epoch GRU sequence models for XJTU-SY and PHM2012. RUL uses `linear_rul_norm`; classification tasks keep `health_state_id` and `early_fault`.

## 2. Config

- input_mode: `feature_sequence`
- sequence.length: 8
- model: `gru`
- max_epochs: 200
- batch_size: 64
- label_source_included: no

## 3. Experiments

| experiment_id | dataset | task | target | feature_count | status |
| --- | --- | --- | --- | --- | --- |
| xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep | XJTU-SY | rul_linear_sequence | linear_rul_norm | 44 | completed |
| xjtu_main_health_gru_sequence_compact_non_label_source_200ep | XJTU-SY | health_state_sequence | health_state_id | 6 | completed |
| xjtu_main_early_gru_sequence_compact_non_label_source_200ep | XJTU-SY | early_fault_sequence | early_fault | 5 | completed |
| phm_official_rul_linear_gru_sequence_compact_non_label_source_200ep | PHM2012 | rul_linear_sequence | linear_rul_norm | 7 | completed |
| phm_official_health_gru_sequence_compact_non_label_source_200ep | PHM2012 | health_state_sequence | health_state_id | 5 | completed |
| phm_official_early_gru_sequence_compact_non_label_source_200ep | PHM2012 | early_fault_sequence | early_fault | 7 | completed |

## 4. Training Completion

| experiment_id | last_epoch | best_epoch | primary_metric | val_primary | test_primary | status |
| --- | --- | --- | --- | --- | --- | --- |
| xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep | 200 | 2 | RMSE | 0.429373 | 0.275009 | completed |
| xjtu_main_health_gru_sequence_compact_non_label_source_200ep | 200 | 7 | WeightedF1 | 0.576007 | 0.375525 | completed |
| xjtu_main_early_gru_sequence_compact_non_label_source_200ep | 200 | 28 | WeightedF1 | 0.618127 | 0.851679 | completed |
| phm_official_rul_linear_gru_sequence_compact_non_label_source_200ep | 200 | 45 | RMSE | 0.254730 | 0.280362 | completed |
| phm_official_health_gru_sequence_compact_non_label_source_200ep | 200 | 1 | WeightedF1 | 0.375393 | 0.417599 | completed |
| phm_official_early_gru_sequence_compact_non_label_source_200ep | 200 | 1 | WeightedF1 | 0.611045 | 0.682715 | completed |

## 5. Findings

- This report intentionally evaluates only the six Step AB 200ep GRU sequence runs; older 50ep runs are not used for the current conclusion.
- XJTU-SY RUL 200ep completed on `linear_rul_norm`; judge it from the 200ep metrics and RUL figures in this report.
- XJTU-SY HealthState and EarlyFault 200ep both completed; EarlyFault is the strongest XJTU classification result in this batch.
- PHM2012 RUL 200ep completed on `linear_rul_norm`; it is not directly comparable to old piecewise-RUL RMSE.
- PHM2012 HealthState and EarlyFault completed; both need visual inspection through the generated confusion matrices.

## 6. Figures

- RUL directories contain `training_curve.png`, `test_true_pred_by_bearing.png`, `test_pred_vs_true.png`, and `test_residuals.png`.
- Classification directories contain `training_curve.png`, `test_confusion_matrix.png`, and `test_class_distribution.png`.

## 7. Decision

- [x] Pass: six 200ep GRU sequence runs completed and were curated.
- [ ] Needs rerun
- [ ] Blocked

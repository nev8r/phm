# Prediction Sanity Audit

## 1. Scope

Step Y-D pauses GUI/demo work and audits completed prediction outputs. No new model training is run.

- Audited experiments: 13
- Alignment rows: 34
- RUL range rows: 18
- Naive comparison rows: 34
- Per-bearing metric rows: 336
- Per-class metric rows: 48

## 2. Direct Answers

1. sample_uid / target 对齐错误：未发现。所有可验证 prediction y_true 均和 labels 表一致。
2. RUL y_pred 越界：存在。
3. clipped RMSE 是否显著改善：部分 split 改善，说明输出范围约束会影响 RMSE 解读。
4. 模型是否打败 naive baseline：存在未打败 naive 的实验/策略，需降级解读。
5. 严重偏离集中在哪些 bearing：见 per_bearing_metrics.csv；RUL 最差条目见下表。
6. 初步归因：若 alignment 通过且 train 明显优于 val/test，优先解释为过拟合、bearing/condition 分布偏移和特征泛化不足；若 clip 改善有限，则不是单纯输出越界问题。
7. 既有报告结论：baseline 排名可以保留为实验记录，但所有 RUL 效果描述必须降级为 early baseline，不应宣称预测质量已经稳定。
8. GUI 状态：暂停最终展示。GUI 可以作为过程演示工具，但不应继续包装为模型效果证明，直到本审计通过验收并完成后续修正。

## 3. Alignment Summary

| experiment_id | split | num_prediction_rows | num_missing_labels | num_duplicate_sample_uid | num_mismatched_targets | alignment_ok |
| --- | --- | --- | --- | --- | --- | --- |
| phm_official_rul_mlp_compact_non_label_source | test | 13025 | 0 | 0 | 0 | yes |
| phm_official_rul_mlp_compact_non_label_source | val | 4330 | 0 | 0 | 0 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | test | 13025 | 0 | 0 | 0 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | val | 4330 | 0 | 0 | 0 | yes |
| xjtu_main_early_mlp_compact_non_label_source | test | 505 | 0 | 0 | 0 | yes |
| xjtu_main_early_mlp_compact_non_label_source | val | 1679 | 0 | 0 | 0 | yes |
| xjtu_main_health_mlp_compact_non_label_source | test | 505 | 0 | 0 | 0 | yes |
| xjtu_main_health_mlp_compact_non_label_source | val | 1679 | 0 | 0 | 0 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | test | 505 | 0 | 0 | 0 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | val | 1679 | 0 | 0 | 0 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | test | 505 | 0 | 0 | 0 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | train | 7032 | 0 | 0 | 0 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | val | 1679 | 0 | 0 | 0 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | test | 505 | 0 | 0 | 0 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | train | 7032 | 0 | 0 | 0 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | val | 1679 | 0 | 0 | 0 | yes |
| y03_xjtu_health_xgboost_compact_non_label_source | test | 505 | 0 | 0 | 0 | yes |
| y03_xjtu_health_xgboost_compact_non_label_source | train | 7032 | 0 | 0 | 0 | yes |
| y03_xjtu_health_xgboost_compact_non_label_source | val | 1679 | 0 | 0 | 0 | yes |
| y05_xjtu_early_xgboost_compact_non_label_source | test | 505 | 0 | 0 | 0 | yes |

## 4. RUL Range Summary

| experiment_id | split | y_pred_min | y_pred_max | clip_rate | raw_RMSE | clipped_RMSE | clip_improves_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | train | -0.043680 | 1.029408 | 0.319966 | 0.046758 | 0.046701 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | test | -0.302538 | 0.858627 | 0.211881 | 0.421645 | 0.408210 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | val | -1.198070 | 17.599522 | 0.157506 | 0.468337 | 0.339076 | yes |
| phm_official_rul_mlp_compact_non_label_source | val | -2.813089 | 9.938951 | 0.142725 | 0.369824 | 0.289969 | yes |
| y07_phm_rul_xgboost_compact_non_label_source | val | -0.040585 | 1.075030 | 0.110162 | 0.291317 | 0.290287 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | test | -0.313650 | 1.365084 | 0.086142 | 0.337661 | 0.336831 | yes |
| phm_official_rul_mlp_compact_non_label_source | test | -2.808210 | 2.792746 | 0.080845 | 0.392475 | 0.335125 | yes |
| y07_phm_rul_xgboost_compact_non_label_source | train | -0.017858 | 1.073524 | 0.073666 | 0.145711 | 0.145616 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | val | -10.337217 | 1.156138 | 0.051817 | 0.339505 | 0.226615 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | test | -0.024940 | 0.982198 | 0.045545 | 0.431454 | 0.431216 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | val | 0.012517 | 1.012197 | 0.038118 | 0.302017 | 0.301954 | yes |
| y07_phm_rul_xgboost_compact_non_label_source | test | 0.026660 | 1.064923 | 0.033935 | 0.357105 | 0.356540 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | train | 0.006893 | 1.000000 | 0.000000 | 0.024701 | 0.024701 | no |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | val | 0.019752 | 1.000000 | 0.000000 | 0.283495 | 0.283495 | no |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | test | 0.011031 | 0.949882 | 0.000000 | 0.399658 | 0.399658 | no |
| y08_phm_rul_random_forest_compact_non_label_source | train | 0.000636 | 1.000000 | 0.000000 | 0.065260 | 0.065260 | no |
| y08_phm_rul_random_forest_compact_non_label_source | val | 0.000642 | 1.000000 | 0.000000 | 0.292971 | 0.292971 | no |
| y08_phm_rul_random_forest_compact_non_label_source | test | 0.000657 | 1.000000 | 0.000000 | 0.337575 | 0.337575 | no |

## 5. Naive Baseline Summary

| experiment_id | naive_strategy | primary_metric | model_test_metric | naive_test_metric | model_beats_naive |
| --- | --- | --- | --- | --- | --- |
| phm_official_rul_mlp_compact_non_label_source | train_mean | RMSE | 0.392475 | 0.330453 | no |
| phm_official_rul_mlp_compact_non_label_source | train_median | RMSE | 0.392475 | 0.325332 | no |
| phm_official_rul_mlp_tuned_compact_non_label_source | train_mean | RMSE | 0.337661 | 0.330453 | no |
| phm_official_rul_mlp_tuned_compact_non_label_source | train_median | RMSE | 0.337661 | 0.325332 | no |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | train_mean | RMSE | 0.421645 | 0.362919 | no |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | train_mean | RMSE | 0.431454 | 0.362919 | no |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | train_mean | RMSE | 0.399658 | 0.362919 | no |
| y07_phm_rul_xgboost_compact_non_label_source | train_mean | RMSE | 0.357105 | 0.330453 | no |
| y07_phm_rul_xgboost_compact_non_label_source | train_median | RMSE | 0.357105 | 0.325332 | no |
| y08_phm_rul_random_forest_compact_non_label_source | train_mean | RMSE | 0.337575 | 0.330453 | no |
| y08_phm_rul_random_forest_compact_non_label_source | train_median | RMSE | 0.337575 | 0.325332 | no |
| y09_phm_health_xgboost_compact_non_label_source | majority_class_0 | WeightedF1 | 0.295374 | 0.328440 | no |
| phm_official_rul_mlp_compact_non_label_source | constant_one | RMSE | 0.392475 | 0.410189 | yes |
| phm_official_rul_mlp_compact_non_label_source | constant_zero | RMSE | 0.392475 | 0.814764 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | constant_one | RMSE | 0.337661 | 0.410189 | yes |
| phm_official_rul_mlp_tuned_compact_non_label_source | constant_zero | RMSE | 0.337661 | 0.814764 | yes |
| xjtu_main_early_mlp_compact_non_label_source | majority_class_0 | WeightedF1 | 0.841682 | 0.213238 | yes |
| xjtu_main_health_mlp_compact_non_label_source | majority_class_0 | WeightedF1 | 0.371101 | 0.213238 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | constant_one | RMSE | 0.421645 | 0.454192 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | constant_zero | RMSE | 0.421645 | 0.768406 | yes |
| xjtu_main_rul_mlp_full_manual_basic_no_reference | train_median | RMSE | 0.421645 | 0.454192 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | constant_one | RMSE | 0.431454 | 0.454192 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | constant_zero | RMSE | 0.431454 | 0.768406 | yes |
| y01_xjtu_rul_xgboost_full_manual_basic_no_reference | train_median | RMSE | 0.431454 | 0.454192 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | constant_one | RMSE | 0.399658 | 0.454192 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | constant_zero | RMSE | 0.399658 | 0.768406 | yes |
| y02_xjtu_rul_random_forest_full_manual_basic_no_reference | train_median | RMSE | 0.399658 | 0.454192 | yes |
| y03_xjtu_health_xgboost_compact_non_label_source | majority_class_0 | WeightedF1 | 0.365121 | 0.213238 | yes |
| y05_xjtu_early_xgboost_compact_non_label_source | majority_class_0 | WeightedF1 | 0.839369 | 0.213238 | yes |
| y07_phm_rul_xgboost_compact_non_label_source | constant_one | RMSE | 0.357105 | 0.410189 | yes |

## 6. Worst Per-Bearing Rows

### RUL

| dataset | task | experiment_id | model_family | split | bearing_id | n | metric | value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHM2012 | rul_tabular | phm_official_rul_mlp_compact_non_label_source | MLP | test | Bearing1_4 | 1428 | RMSE | 0.697326 |
| PHM2012 | rul_tabular | phm_official_rul_mlp_compact_non_label_source | MLP | test | Bearing2_6 | 701 | RMSE | 0.653506 |
| PHM2012 | rul_tabular | y07_phm_rul_xgboost_compact_non_label_source | xgboost_regressor | test | Bearing2_6 | 701 | RMSE | 0.600752 |
| PHM2012 | rul_tabular | y08_phm_rul_random_forest_compact_non_label_source | random_forest_regressor | test | Bearing2_6 | 701 | RMSE | 0.580421 |
| XJTU-SY | rul_tabular | y02_xjtu_rul_random_forest_full_manual_basic_no_reference | random_forest_regressor | test | Bearing3_5 | 114 | RMSE | 0.578098 |
| XJTU-SY | rul_tabular | y01_xjtu_rul_xgboost_full_manual_basic_no_reference | xgboost_regressor | test | Bearing3_5 | 114 | RMSE | 0.576185 |
| XJTU-SY | rul_tabular | xjtu_main_rul_mlp_full_manual_basic_no_reference | MLP | test | Bearing3_5 | 114 | RMSE | 0.575838 |
| XJTU-SY | rul_tabular | y01_xjtu_rul_xgboost_full_manual_basic_no_reference | xgboost_regressor | test | Bearing1_5 | 52 | RMSE | 0.482708 |
| PHM2012 | rul_tabular | y07_phm_rul_xgboost_compact_non_label_source | xgboost_regressor | test | Bearing1_7 | 2259 | RMSE | 0.469789 |
| PHM2012 | rul_tabular | phm_official_rul_mlp_tuned_compact_non_label_source | MLP | test | Bearing1_7 | 2259 | RMSE | 0.469228 |

### Classification

| dataset | task | experiment_id | model_family | split | bearing_id | n | metric | value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PHM2012 | early_fault_tabular | y11_phm_early_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing2_6 | 701 | WeightedF1 | 0.000782 |
| PHM2012 | health_state_tabular | y09_phm_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing2_6 | 701 | WeightedF1 | 0.017093 |
| XJTU-SY | health_state_tabular | xjtu_main_health_mlp_compact_non_label_source | MLP | test | Bearing3_5 | 114 | WeightedF1 | 0.040508 |
| XJTU-SY | health_state_tabular | y03_xjtu_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing3_5 | 114 | WeightedF1 | 0.043708 |
| PHM2012 | early_fault_tabular | y11_phm_early_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing1_6 | 2448 | WeightedF1 | 0.187140 |
| PHM2012 | health_state_tabular | y09_phm_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing2_7 | 230 | WeightedF1 | 0.188116 |
| XJTU-SY | health_state_tabular | xjtu_main_health_mlp_compact_non_label_source | MLP | test | Bearing1_5 | 52 | WeightedF1 | 0.189865 |
| PHM2012 | health_state_tabular | y09_phm_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing1_7 | 2259 | WeightedF1 | 0.195811 |
| XJTU-SY | health_state_tabular | y03_xjtu_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing1_5 | 52 | WeightedF1 | 0.204212 |
| PHM2012 | health_state_tabular | y09_phm_health_xgboost_compact_non_label_source | xgboost_classifier | test | Bearing1_6 | 2448 | WeightedF1 | 0.206009 |

## 7. Generated Figures

- `figures/*_val_true_pred_by_bearing.png`
- `figures/*_test_true_pred_by_bearing.png`

## 8. Decision

- [x] Step Y-D audit artifacts generated
- [x] No new training was run
- [x] GUI remains paused as a model-quality claim
- [ ] Ready to resume final GUI/video narrative

# 工业轴承设备剩余寿命预测系统的实现：指标驱动实验结果说明与 RUL 改进任务书

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | 指标驱动实验结果说明与 RUL 改进任务书 |
| 文档编号 | SE-PHM-FINAL-28 |
| 文档阶段 | 结题补充 |
| 项目名称 | 工业轴承设备剩余寿命预测系统的实现 |
| 课程 | 中国科学技术大学软件学院《软件工程》 |
| 指导老师 | zjf |
| 小组成员 | zyj、cyy、zdh、zy |
| 文档负责人 | zyj |
| 参与编写 | cyy、zdh、zy |
| 版本 | V2.0 |
| 修订日期 | 2026-06-17 |
| 归档形式 | Markdown 源文件、PDF、DOCX |
| 内容基线 | 已执行的 tsfresh/sktime/RULSurv 指标证据、答辩边界和后续工作 |

## 修订记录

| 版本 | 日期 | 编写人 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2026-06-16 | 项目组 | 根据 `next goal.md` 整理下一阶段指标驱动 RUL 改进任务书 |
| V2.0 | 2026-06-17 | 项目组 | 同步已完成实验结果，删除未来计划模板和空表 |

## 1. 文档目的

本文档用于说明 #2 后已经完成的指标驱动补充实验，以及这些实验在结题答辩中应如何表述。当前重点不是继续扩大模型数量，而是把手工特征、tsfresh、sktime、RULSurv 和 external SOTA evidence 放到同一边界下解释。

统一结论如下：

> 项目已完成 tsfresh train-only 特征筛选、tsfresh RUL baseline、sktime Rocket/TimeSeriesForest baseline、strict same-config repeated seed rerun、RULSurv RSF port 以及 external SOTA source pin / dependency probe evidence。结题材料可以引用这些证据，但不能把 external SOTA probe 写成本地复现完成，不能把 tsfresh 写成核心性能突破，也不能把 RULSurv held-out 迁移写成自然泛化完全解决。

## 2. 当前数据划分与防泄漏协议

本轮指标驱动实验使用 XJTU-SY condition 1 的固定 held-out bearing split：

| 字段 | 取值 |
| --- | --- |
| dataset_name | XJTU-SY |
| condition_name | condition_1_35Hz12kN |
| train_entities | Bearing1_1, Bearing1_2, Bearing1_4, Bearing1_5 |
| test_entities | Bearing1_3 |
| split_name | train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3 |
| target | positive-RUL snapshots / held-out RUL regression |

防泄漏原则：

- 先按轴承划分 train/test；
- tsfresh selector、scaler 和模型只在 train bearings 上 fit；
- Bearing1_3 只用于最终 transform/predict/evaluate；
- summary 与 predictions 均保留 dataset、condition、split、seed 和 prediction_count 字段。

## 3. tsfresh 已执行结果

tsfresh 的定位是自动特征分析旁证，不是替代主线深度模型的核心突破。

已生成证据：

```text
docs/reproduction-evidence/tsfresh_feature_relevance_summary.csv
docs/reproduction-evidence/tsfresh_feature_relevance_summary.md
docs/reproduction-evidence/tsfresh_rul_baseline_summary.csv
docs/reproduction-evidence/tsfresh_rul_baseline_predictions.csv
```

关键结果：

| 方法 | run_count | mean normalized RMSE | 说明 |
| --- | ---: | ---: | --- |
| manual 19 features + RandomForest | 3 | 0.345365 | 手工特征传统模型 baseline |
| tsfresh selected features + RandomForest | 3 | 0.315629 | 对手工 RF 有一定改进 |

特征相关性边界：

- top feature `vertical__maximum` correlation 约 `0.200994`；
- `vertical__absolute_maximum` correlation 约 `0.125944`；
- 后续 selected features 中不少 correlation 只有 `0.04` 左右，p-value 较高；
- 多个能量、方差、RMS 类候选特征分数为 0；
- 因此结题材料应说“tsfresh 相关性整体偏弱，只能作为自动特征分析旁证”，不能说“tsfresh 发现了强退化特征”。

## 4. sktime 已执行结果

sktime 已作为时间序列回归 baseline 接入，使用同一 held-out split 与 RUL 标签。

已生成证据：

```text
docs/reproduction-evidence/sktime_rul_baseline_summary.csv
docs/reproduction-evidence/sktime_rul_baseline_predictions.csv
```

关键结果：

| 方法 | input_format | run_count | mean normalized RMSE | 说明 |
| --- | --- | ---: | ---: | --- |
| sktime RocketRegressor | sktime_3d_panel_numpy | 3 | 0.263706 | 当前优于 tsfresh RandomForest |
| sktime TimeSeriesForestRegressor | sktime_3d_panel_numpy | 3 | 0.315919 | 与 tsfresh RandomForest 接近 |

答辩口径：

> sktime RocketRegressor 可以作为强 baseline 参考。它说明时间序列特征变换路线在 held-out Bearing1_3 split 上优于本轮 tsfresh RF，但仍只是项目 split 的 baseline，不等同于外部 SOTA。

## 5. strict repeated seed 已执行结果

已生成证据：

```text
docs/reproduction-evidence/strict_repeated_seed_summary.csv
docs/reproduction-evidence/strict_repeated_seed_config.json
```

关键结果：

| 方法 | seeds | epochs | mean normalized RMSE | config_hash |
| --- | ---: | ---: | ---: | --- |
| XLSTM-Transformer | 3 | 50 | 0.157007 | 7ee11220b98a7cc1 |
| Feature-Transformer | 3 | 50 | 0.186688 | 7ee11220b98a7cc1 |

该结果解决了“没有重复种子证据”的旧口径，但仍不等同于外部作者源码级复现，也不能覆盖所有数据集和工况。

## 6. RULSurv 已执行结果

RULSurv 必须分成两个口径解释。

已生成证据：

```text
docs/reproduction-evidence/rulsurv_rsf_port/rulsurv_rsf_port_summary.csv
docs/reproduction-evidence/rulsurv_rsf_port/rulsurv_rsf_port_metrics.csv
docs/reproduction-evidence/rulsurv_rsf_port/rulsurv_rsf_port_predictions.csv
docs/reproduction-evidence/rulsurv_rsf_port/rulsurv_rsf_port_config.json
```

关键结果：

| 协议 | run_count | mean true MAE | 状态 | 解释 |
| --- | ---: | ---: | --- | --- |
| original 25% censored row-level CV | 3 | 6.926416 min | PROTOCOL_PASS | RULSurv-compatible 原协议近似复现 |
| project Bearing1_3 held-out migration | 3 | 14.307856 min | MIGRATION_PASS | 本项目固定 split 迁移结果 |

必须保留的边界：

- row-level CV 可能把同一 bearing 的不同时间点分到不同 fold，因此不能等同于 held-out bearing 泛化；
- project held-out pass 依赖 `project_holdout_survival_probability=0.25`，即 `survival_probability=0.25` 的固定 conservative survival quantile 保守解码；
- 这说明 RULSurv RSF 有迁移潜力，但不能说成原始 RULSurv 自然泛化问题已经完全解决。

## 7. external SOTA 已执行证据

外部 SOTA 没有本地重跑完成。当前完成的是 source pin、依赖 probe 和下一步复现路径。

已生成证据：

```text
docs/reproduction-evidence/external_sota_attempts.csv
docs/reproduction-evidence/external_sota_attempts/autorul-pronostia-femto-rmse.txt
docs/reproduction-evidence/external_sota_attempts/gnn-benchmark-phm2012-fc-stgnn.txt
docs/reproduction-evidence/external_sota_attempts/weibull-kiml-femto-rmse.txt
```

| route | 当前状态 | 失败边界 |
| --- | --- | --- |
| AutoRUL / auto-sktime | source pin 成功，dependency probe 失败 | 当前环境缺 `autosklearn`，尚未 materialize PRONOSTIA benchmark layout |
| GNN_RUL_Benchmarking | source pin 成功，dependency probe 失败 | 当前环境缺 `torch_geometric`，尚未生成其 PHM2012 preprocessed split |
| Weibull KIML | source pin 成功，dependency probe 失败 | 当前环境缺 survival/deep-learning 依赖，尚未 stage FEMTO make workflow |

答辩口径：

> 外部 SOTA 已完成代码源锁定、依赖 probe 和可复现路径设计，但 AutoRUL / GNN / Weibull KIML 尚未在本地跑出指标，因此只能作为后续强基线目标，不能作为已完成本地复现结果。

## 8. 统一结果表

| 方法 | 数据 / split | run_count | mean normalized RMSE / MAE | 结论 |
| --- | --- | ---: | ---: | --- |
| manual 19 features + RandomForest | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.345365 | 传统 baseline |
| tsfresh selected features + RandomForest | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.315629 | 对手工 RF 有补充，但相关性整体偏弱 |
| sktime RocketRegressor | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.263706 | 当前补充 baseline 中最好 |
| sktime TimeSeriesForestRegressor | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.315919 | 与 tsfresh RF 接近 |
| XLSTM-Transformer strict rerun | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.157007 | 50 epoch 同配置重复证据 |
| Feature-Transformer strict rerun | XJTU-SY Bearing1_3 held-out | 3 | NRMSE 0.186688 | 50 epoch 同配置重复证据 |
| RULSurv RSF row-level CV | XJTU-SY condition 1 | 3 | MAE 6.926416 min | PROTOCOL_PASS |
| RULSurv RSF held-out migration | XJTU-SY Bearing1_3 held-out | 3 | MAE 14.307856 min | MIGRATION_PASS，依赖 0.25 保守解码 |

## 9. 结题材料推荐话术

可以放入 PPT 的短版话术：

```text
外部 SOTA：AutoRUL / GNN / Weibull KIML 已完成 source pin 和依赖 probe，但尚未在本地跑出指标，不能宣称这些路线已复现完成。

tsfresh：已完成 train-only 特征筛选和 held-out baseline，但本次特征相关性整体偏弱，只能作为自动特征分析旁证，不是核心性能突破。

RULSurv：row-level 协议复现达到 target；held-out Bearing1_3 pass 依赖 survival_probability=0.25 的保守解码，是项目迁移策略，不能说成原始 RULSurv 自然泛化已经完全解决。
```

## 10. 后续工作

1. 为 AutoRUL 建立独立环境或容器，重跑 `femto_bearing` 并输出真实本地指标。
2. 为 GNN_RUL_Benchmarking 生成其 PHM2012 preprocessing split，复跑 FC-STGNN。
3. 为 Weibull KIML stage FEMTO 数据和 make workflow，补齐可靠性先验路线指标。
4. 扩展 RULSurv RSF port 到更多 XJTU 工况，验证 `survival_probability=0.25` 保守解码是否稳定。
5. 将 PHM2012 温度字段纳入特征融合实验，并扩展多 seed、多工况和更大样本规模。

## 11. 本轮完成定义

本轮完成标准是：

- 已把 tsfresh/sktime/RULSurv/external SOTA 的真实完成度同步进结题材料；
- 已删除“只做未来计划、不提交实验 CSV”的过期口径；
- 已将 taskbook 改成当前结果说明和后续工作清单；
- 已明确外部 SOTA 未重跑、tsfresh 相关性整体偏弱、RULSurv held-out 为 `survival_probability=0.25` 保守解码迁移结果；
- 课程归档材料应以本文件、`docs/project-owner/08_指标驱动实验结果说明.md` 和 `docs/reproduction-evidence/README.md` 的一致口径为准。

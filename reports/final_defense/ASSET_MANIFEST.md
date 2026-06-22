# 最终答辩资产清单

## 1. 目的

本清单用于整理最终论文、答辩 PPT 和演示材料可引用的资产。Step AD 只做资料收敛与展示规划，不新增训练，不改动已有实验结论。

后续文档统一使用以下表述：

- 论文复现实验
- 本文方法实验
- 对照模型实验
- GRU 序列基线
- CAM-LSTM 复现模型

## 2. 可直接进入正文的核心资产

| 资产组 | 位置 | 内容 | 正文用途 | 注意事项 |
| --- | --- | --- | --- | --- |
| 数据集与任务说明 | `user-guide/` | XJTU-SY 和 PHM2012 数据集说明与加载方式 | 数据集章节 | 对齐官方说明，不展开额外预处理假设 |
| 特征分析总报告 | `reports/feature_analysis/FEATURE_ANALYSIS_REPORT.md` | 特征筛选、任务适配性、跨数据集讨论 | 特征工程章节 | 适合解释为什么选 compact/full 特征集 |
| 特征分析 LaTeX 文档 | `reports/feature_analysis/latex/` | 中文技术背景、名词解释、特征指标说明 | 论文正文基础 | 后续可整合进最终中文论文 |
| 推荐特征表 | `reports/feature_analysis/summary/recommended_features.csv` | 每个数据集和任务的推荐特征 | 方法与实验设置 | 需要强调是由特征分析驱动 |
| XJTU-SY 特征图 | `reports/feature_analysis/xjtu_sy/all_conditions_bearing_index_manual_basic/figures/` | RUL 相关性、健康状态箱线图、早期故障效应、推荐矩阵 | 特征分析结果 | 适合放 2 到 3 张代表图 |
| PHM2012 特征图 | `reports/feature_analysis/phm2012/manual_basic/figures/` | RUL 相关性、健康状态箱线图、早期故障效应、推荐矩阵 | 特征分析结果 | 与 XJTU-SY 做并列表达 |
| GRU 200ep 指标汇总 | `reports/sequence_baseline_results/gru_sequence_200ep_metrics.csv` | 2 个数据集乘 3 个任务的 GRU 结果 | 本文方法实验主结果 | RUL 使用 `linear_rul_norm` |
| GRU 200ep 报告 | `reports/sequence_baseline_results/02_gru_sequence_200ep_batch.md` | 全任务 GRU 训练结论 | 本文方法实验章节 | 作为最终模型结果的主文档 |
| XJTU-SY GRU RUL 图 | `reports/final_defense/assets/gru_sequence/` | true/pred by bearing、pred-vs-true、residuals | RUL 结果展示 | 可以展示预测曲线，但要保留泛化风险说明 |
| XJTU-SY GRU EarlyFault 图 | `reports/final_defense/assets/gru_sequence/` | 混淆矩阵和类别分布 | 早期故障检测结果展示 | 是当前最稳的正向分类结果 |
| PHM2012 GRU 图 | `reports/sequence_baseline_results/phm_official_*_200ep/figures/` | RUL 和分类任务图 | PHM2012 结果展示 | HealthState 和 EarlyFault 要谨慎解读 |
| 非 MLP 对照模型 | `reports/non_mlp_baseline_results/` | XGBoost 与 RandomForest 结果和诊断图 | 对照模型实验 | 主要说明树模型并非整体优于 GRU |
| 预测质量排查 | `reports/prediction_audit/` | 对齐检查、越界检查、naive baseline 对比、per-bearing 诊断 | 实验可信度与局限性 | 只作为质量控制与风险说明，不包装成正向效果 |
| 演示视频 | `reports/demo_videos/video/` | 两个 50ep 加速训练过程视频 | 系统演示章节 | 视频是过程演示，最终结论仍引用 200ep 结果 |

## 3. 论文复现实验资产

论文复现实验素材已归档到：

```text
reports/final_defense/assets/paper_reproduction/
```

| 复现实验 | 代表文件 | 指标 | 用途 |
| --- | --- | --- | --- |
| PHM2012 RUL 的 CAM-LSTM 复现模型 | `phm2012_rul_prediction_by_bearing.png`、`phm2012_rul_prediction_curve.png`、`phm2012_rul_reproduction_metrics.json` | test RMSE = 0.146737，MAE = 0.105007，R2 = 0.733183 | 说明已尝试复现论文模型，并作为 RUL 对照 |
| XJTU-SY 故障诊断复现模型 | `xjtu_fault_confusion_matrix.png`、`xjtu_fault_reproduction_metrics.json` | test Accuracy = 0.997072，WeightedF1 = 0.997072 | 说明故障诊断任务上论文复现模型表现很强 |
| 复现模型架构图 | `rul_model_architecture.png`、`fault_model_architecture.png` | 架构示意 | PPT 中解释复现实验模型结构 |
| 复现实验 benchmark 图 | `rul_reproduction_benchmark.png`、`fault_reproduction_benchmark.png` | 不同模型对照 | 可放附录或答辩备选页 |

这些资产后续可以命名为“论文复现实验”，不要把它们和本文重新整理后的三任务实验混写成同一套实验。

## 4. 不作为正向结果展示的资产

| 资产 | 原因 | 建议用途 |
| --- | --- | --- |
| 旧的 `piecewise_rul_norm` RUL 曲线 | 已决定 RUL 统一改为 `linear_rul_norm` | 只在“问题修正”或附录中说明 |
| 早期 MLP 和表格模型 RUL 正向曲线 | 预测质量排查显示部分模型未稳定超过 naive baseline | 作为排查证据，不放主结果页 |
| 静态截图轮播类视频 | 不能证明训练过程动态变化 | 不作为最终演示视频 |
| 含“训练曲线作为结果图”的视频页 | 用户已明确不希望结果页展示训练曲线 | 仅保留真正的结果图展示 |
| 资源不足导致未完成的 full-size tsfresh 对照 | 没有产出有效 feature ranking | 只作为工程限制说明 |

## 5. 正文优先展示顺序

1. 数据集与任务定义：XJTU-SY、PHM2012、RUL、HealthState、EarlyFault。
2. 标签构造：RUL 使用 `linear_rul_norm`，HealthState 和 EarlyFault 为退化阶段伪标签。
3. 特征分析：从推荐矩阵、RUL top features、箱线图、早期故障效应说明特征选择。
4. 本文方法实验：GRU 序列基线全任务结果。
5. 对照模型实验：XGBoost、RandomForest、MLP 与 GRU 的比较。
6. 论文复现实验：CAM-LSTM 复现模型和 XJTU-SY 故障诊断复现模型。
7. 质量排查：对齐正确，但 RUL 泛化仍有风险。
8. 系统演示：两个加速训练视频作为工程演示。

## 6. 当前资产状态

- [x] 特征分析报告和图表齐全。
- [x] GRU 序列基线 200ep 六个任务结果齐全。
- [x] XGBoost 与 RandomForest 对照结果齐全。
- [x] 预测质量排查报告齐全。
- [x] 两个中文加速训练视频通过验收。
- [x] 两组论文复现实验素材已归档到最终答辩资产目录。
- [ ] 最终中文 LaTeX 论文待 Step AE 编写。
- [ ] USTC 风格 Beamer PPT 待 Step AF 编写。

# 最终结果筛选规则

## 1. 筛选目标

最终论文和答辩材料只展示能支撑主线叙事的结果。所有结果按照以下优先级筛选：

1. 目标定义正确，尤其是 RUL 必须使用 `linear_rul_norm`。
2. 实验可复现，结果文件和图像文件完整。
3. 能回答论文问题：特征是否有效、模型是否能利用时序、对照模型表现如何、局限在哪里。
4. 图像适合非本领域听众理解，不展示会造成误解的训练过程图。

## 2. 主结果选择

| 数据集 | 任务 | 推荐展示模型 | 推荐图 | 展示结论 |
| --- | --- | --- | --- | --- |
| XJTU-SY | RUL | GRU 序列基线 | true/pred by bearing、pred-vs-true、residuals | 使用全量人工特征并排除标签源特征后，GRU 能形成连续寿命预测，但仍需说明跨轴承泛化难度 |
| XJTU-SY | HealthState | GRU 序列基线 | confusion matrix、class distribution | 作为退化阶段分类任务展示，结论应保守 |
| XJTU-SY | EarlyFault | GRU 序列基线 | confusion matrix、class distribution | 当前最适合作为正向分类结果，test WeightedF1 = 0.851679 |
| PHM2012 | RUL | GRU 序列基线 | true/pred by bearing、pred-vs-true、residuals | 使用 `linear_rul_norm` 后可展示为主线 RUL 结果，test RMSE = 0.280362 |
| PHM2012 | HealthState | GRU 序列基线 | confusion matrix、class distribution | 可展示，但要说明伪标签任务难度较大 |
| PHM2012 | EarlyFault | GRU 序列基线 | confusion matrix、class distribution | 可展示，test WeightedF1 = 0.682715 |

GRU 200ep 六个任务的指标如下：

| 数据集 | 任务 | 指标 | 测试集结果 |
| --- | --- | --- | ---: |
| XJTU-SY | RUL | RMSE | 0.275009 |
| XJTU-SY | HealthState | WeightedF1 | 0.375525 |
| XJTU-SY | EarlyFault | WeightedF1 | 0.851679 |
| PHM2012 | RUL | RMSE | 0.280362 |
| PHM2012 | HealthState | WeightedF1 | 0.417599 |
| PHM2012 | EarlyFault | WeightedF1 | 0.682715 |

## 3. 特征分析选择

特征分析不只展示分数表，而是要回答“为什么这个特征有用”。

| 图表 | 展示用途 | 推荐位置 |
| --- | --- | --- |
| `feature_recommendation_matrix.png` | 说明每个任务推荐哪些特征族 | 论文特征分析章节和 PPT 方法页 |
| `rul_top_features.png` | 说明 RUL 与特征单调趋势或相关性 | 论文实验分析章节 |
| `health_state_boxplots.png` | 说明健康阶段之间的特征可分性 | PPT 特征页 |
| `early_fault_effects.png` | 说明早期故障任务中哪些特征变化明显 | PPT 结果页 |
| `feature_score_heatmap.png` | 汇总不同任务分数 | 附录或备选页 |
| `degradation_score_heatmap.png` | 解释退化趋势和 prognosability | 附录或正文补充 |

正文优先展示 XJTU-SY 和 PHM2012 各 2 到 3 张图。完整分数表放附录。

## 4. 对照模型选择

对照模型实验用于说明“不是所有传统表格模型都能稳定解决该问题”。

| 模型 | 展示方式 | 结论口径 |
| --- | --- | --- |
| XGBoost | 指标表、feature importance、pred-vs-true 或 confusion matrix | 可作为强表格基线，但 RUL 有过拟合和分布偏移风险 |
| RandomForest | 指标表、feature importance、pred-vs-true 或 confusion matrix | PHM2012 RUL 上优于默认 MLP，但不是最终主线模型 |
| MLP | 只保留基线比较表 | 作为表格神经网络基线，不作为最终效果主图 |

对照模型的核心结论：

- XJTU-SY EarlyFault 中，XGBoost test WeightedF1 = 0.839369，RandomForest test WeightedF1 = 0.837047，接近 GRU 序列基线。
- PHM2012 RUL 中，RandomForest test RMSE = 0.337575，优于默认 MLP，但仍弱于 GRU 序列基线。
- RUL 对照模型需要结合 naive baseline 排查结果谨慎表达。

## 5. 论文复现实验选择

| 复现实验 | 是否进入正文 | 展示方式 | 结论口径 |
| --- | --- | --- | --- |
| PHM2012 RUL 的 CAM-LSTM 复现模型 | 是 | 模型结构、预测曲线、指标表 | 作为论文复现实验对照，不直接替代本文三任务框架 |
| XJTU-SY 故障诊断复现模型 | 是 | 混淆矩阵、指标表 | 说明在故障诊断任务上复现模型表现强 |
| benchmark 汇总图 | 可选 | 附录或备选页 | 用于回答对照模型问题 |

论文复现实验建议放在“复现实验对照”章节，而不是放进本文三任务主结果表。

## 6. 不进入正向展示的结果

| 结果 | 处理方式 | 原因 |
| --- | --- | --- |
| `piecewise_rul_norm` RUL 结果 | 不作为最终 RUL 结果 | RUL 已统一为 `linear_rul_norm` |
| 早期 RUL 图中出现长时间常数阶段的曲线 | 不作为最终效果图 | 容易误导听众以为寿命标签或模型输出异常 |
| 未通过资源限制的 full-size tsfresh 对照 | 只在局限性中说明 | 没有完整有效结果 |
| 只含截图轮播的视频 | 不进入最终演示 | 不能展示训练过程变化 |
| 结果页中展示训练曲线的视频版本 | 不进入最终演示 | 用户已确认结果页只放结果图 |

## 7. 质量排查结论的使用方式

预测质量排查用于增强可信度，而不是削弱主线：

- 对齐检查通过：sample_uid 与 label 对齐未发现错误。
- RUL 存在输出越界和未打败 naive baseline 的早期模型。
- 因此，旧 RUL baseline 只能作为探索过程，不能作为最终正向结论。
- GRU 序列基线使用 `linear_rul_norm`，作为当前主线结果。
- 答辩时要主动说明：RUL 仍是最难任务，未来需要物理特征、跨工况训练和更强序列模型。

## 8. 最终表达口径

最终材料建议使用以下判断：

1. 本项目完成了从数据加载、特征分析、标签构造、任务数据集、模型训练到演示视频的完整闭环。
2. 特征分析证明人工振动特征对 RUL、健康阶段和早期故障任务有不同适配性。
3. GRU 序列基线是当前本文方法实验的主线结果。
4. XGBoost 和 RandomForest 是对照模型实验，帮助解释表格模型的强弱边界。
5. CAM-LSTM 复现模型用于体现论文复现实验能力。
6. RUL 泛化仍是主要限制，不能夸大为工业级可部署预测精度。

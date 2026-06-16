# 工业轴承设备剩余寿命预测系统的实现：指标驱动实验结果说明与 RUL 改进任务书

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | 指标驱动实验结果说明与 RUL 改进任务书 |
| 文档编号 | SE-PHM-FINAL-28 |
| 文档阶段 | 结题后续改进 |
| 项目名称 | 工业轴承设备剩余寿命预测系统的实现 |
| 课程 | 中国科学技术大学软件学院《软件工程》 |
| 指导老师 | zjf |
| 小组成员 | zyj、cyy、zdh、zy |
| 文档负责人 | zyj |
| 参与编写 | cyy、zdh、zy |
| 版本 | V1.0 |
| 修订日期 | 2026-06-16 |
| 归档形式 | Markdown 源文件、PDF、DOCX |
| 内容基线 | 指标驱动实验协议、tsfresh/sktime 接入方案、验收标准和风险边界 |

## 修订记录

| 版本 | 日期 | 编写人 | 说明 |
| --- | --- | --- | --- |
| V1.0 | 2026-06-16 | 项目组 | 根据 `next goal.md` 整理下一阶段指标驱动 RUL 改进任务书 |

## 1. 文档目的

本文档用于指导后续同学把项目从“系统架构完成度”推进到“RUL 预测指标提升”。它不是本轮新增代码实现说明，而是下一阶段的设计建议、实验方案、验收标准、风险边界，以及后续实验结果说明的填写模板。

下一阶段目标是：

> 在相同 train/test 轴承划分、相同 RUL 标签、相同评价指标下，比较手工 19 维特征、tsfresh 自动特征、sktime 时间序列回归 baseline 与现有深度模型，找出指标更优的特征和模型组合。

本轮不在本地实现 tsfresh、sktime 或新模型代码，不提交新的实验结果 CSV，也不声称新增方法已经优于现有论文复现模型。

## 2. 当前问题诊断

当前项目主线为：

```text
loader
  -> preprocessing
  -> 19 维手工时频域特征
  -> FeatureSequenceRulLabeler
  -> CNN-LSTM-AM / xLSTM-Transformer
  -> RMSE / NormalizedRMSE / R2 / Score
```

该链路已经具备数据集语义、RUL 标签、19 维可解释特征、两篇论文模型和 formal 50 epoch 真实训练指标输出。但目前更像是在证明系统架构完整，而不是系统性证明“哪种特征和模型能把 RUL 指标做上去”。

后续应调整为：

```text
以 RUL 指标为目标
  -> 保留现有系统架构
  -> 把特征提取、模型训练封装为可插拔 backend
  -> 接入 tsfresh / sktime / sklearn / xgboost 等方案
  -> 在相同数据划分下公平比较
  -> 用指标选择更优路线
```

## 3. 数据概念必须统一

后续开发不能把工况、轴承、快照和训练样本混在一起。推荐统一数据层级为：

```text
数据集 dataset
  -> 工况 condition
    -> 轴承 bearing
      -> 快照 snapshot / csv 文件
        -> 振动通道 channel
          -> 滑动窗口 window
            -> 19 维特征
              -> 特征序列
                -> RUL 标签
```

一句话口径：

> 工况管实验条件，轴承管一次寿命实验，csv 管某一时刻的振动快照，模型吃的是连续快照提取出的特征序列。

XJTU-SY 数据特点：

| 要点 | 说明 |
| --- | --- |
| 工况 | 3 个工况 |
| 轴承 | 每个工况 5 个轴承，共 15 个 run-to-failure 轴承 |
| 快照 | 每个 CSV 是一个振动快照，间隔约 1 分钟 |
| 点数 | 每个快照 32768 个点 |
| 通道 | 主要使用水平、垂直振动通道 |

PHM2012 / FEMTO 数据特点：

| 要点 | 说明 |
| --- | --- |
| 工况 | 3 个工况 |
| learning set | 每个工况常用 2 个完整寿命训练轴承 |
| challenge/test | 存在截断语义，不能简单把最后文件当失效点 |
| 快照 | `acc_*.csv` 是振动快照，`temp_*.csv` 是温度文件 |
| 当前主线 | 主模型只使用振动特征，温度由 loader 保留但未进入主线模型 |

## 4. 当前复现状态与边界

正式复现实验已经具备真实数据、50 epoch 和指标输出，但仍属于真实数据抽样复现，不是作者源码级或全量数据级复现。

| 实验 | 数据划分 | 当前边界 |
| --- | --- | --- |
| CNN-LSTM-AM | XJTU-SY condition 1：Bearing1_1、1_2、1_4、1_5 训练，Bearing1_3 测试；PHM2012 condition 1：Bearing1_1、1_2 训练，Bearing1_3 测试 | 只覆盖两个 condition 1 对照 |
| xLSTM-Transformer | XJTU-SY 三工况测试 `*_3`；PHM2012 三工况测试 `*_3` | 覆盖六工况，但 PHM2012 condition 2 与 Score 仍有明显差距 |
| 抽样规模 | 每轴承按时间均匀抽样 96 个快照 | 不等于全量快照训练 |
| 训练轮数 | 50 epoch | 已脱离 smoke/demo，但仍不是论文全量训练 |

推荐正式表述：

> 当前完成的是正式真实数据抽样复现和论文指标对照。受算力和时间限制，每轴承采用 96 个快照的时间均匀抽样，并完成 50 epoch 训练。

## 5. 统一实验协议

后续所有实验必须先确定数据划分，再运行特征、模型和评价，禁止每个模型各自拆数据。

推荐 `DataSplit` 配置字段：

```yaml
dataset_name: XJTU-SY
condition_name: condition_1_35Hz12kN
train_entities:
  - Bearing1_1
  - Bearing1_2
  - Bearing1_4
  - Bearing1_5
test_entities:
  - Bearing1_3
max_samples_per_entity: 96
target_mode: entity_relative
sequence_length: 10
```

必须统一输出以下配置字段：

| 字段 | 含义 |
| --- | --- |
| `dataset_name` | 数据集名称 |
| `condition_name` | 工况 |
| `train_entities` / `test_entities` | 训练/测试轴承 |
| `max_samples_per_entity` | 每轴承抽样快照数 |
| `target_mode` | RUL 标签尺度，如 `entity_relative` |
| `sequence_length` | 特征序列长度 |
| `feature_backend` | 特征后端 |
| `model_backend` | 模型后端 |

## 6. 可插拔 backend 设计

建议后续架构组织为：

```text
DataSplit
  -> FeatureBackend
  -> ModelBackend
  -> Evaluator
  -> ExperimentReporter
```

### 6.1 FeatureBackend

特征后端负责把原始轴承数据转换为模型输入。建议至少规划三类：

| 后端 | 作用 |
| --- | --- |
| `HandcraftedFeatureBackend` | 当前 19 维手工特征 baseline |
| `TSFreshFeatureBackend` | 自动特征提取、相关性排序和特征选择 |
| `SktimeFeatureBackend` | 面向 sktime panel 或 Catch22/TSFresh transformer 的格式转换 |

建议接口语义：

```python
class FeatureBackend:
    name: str

    def fit(self, train_entities, y_train):
        ...

    def transform(self, entities):
        ...

    def fit_transform(self, train_entities, y_train):
        ...
```

红线要求：

- scaler、selector、feature filter 只能在训练轴承上 `fit`；
- 测试轴承只能 `transform`；
- 禁止把 train 和 test 合在一起做特征选择；
- 输出必须保留 `metadata_frame`，记录样本来自哪个轴承和快照范围。

### 6.2 ModelBackend

模型后端负责训练和预测。建议至少规划：

| 后端 | 作用 |
| --- | --- |
| `PyTorchModelBackend` | 现有 CNN-LSTM-AM、xLSTM-Transformer |
| `SklearnRegressorBackend` | Ridge、RandomForest、XGBoost 等传统回归 |
| `SktimeRegressorBackend` | RocketRegressor、TimeSeriesForestRegressor 等时间序列回归 |

建议接口语义：

```python
class ModelBackend:
    name: str

    def fit(self, X_train, y_train):
        ...

    def predict(self, X_test):
        ...
```

第一阶段不要求把所有模型工程化得很重，但必须统一输出 `predictions.csv`、`metrics.json`、`comparison_metrics.csv`。

### 6.3 ExperimentReporter

每个实验至少输出：

| 字段 | 说明 |
| --- | --- |
| `experiment_name` | 实验名称 |
| `feature_backend` / `model_backend` | 特征与模型路线 |
| `rmse` / `normalized_rmse` / `r2` | 第一优先级指标 |
| `huang_rul_score` / `phm2012_score` | RUL 惩罚指标 |
| `prediction_count` | 预测样本数 |
| `fit_seconds` / `predict_seconds` | 训练与预测耗时 |
| `notes` | 边界、异常或解释 |

建议落盘：

```text
outputs/metric_driven_experiments/<run_name>/
  comparison_metrics.csv
  predictions.csv
  metrics.json
  config.json
```

提交摘要时使用：

```text
docs/reproduction-evidence/metric_driven_comparison_summary.csv
```

## 7. tsfresh 接入方案

tsfresh 的定位不是直接替代主线模型，而是作为自动特征分析工具、传统机器学习输入特征和 19 维手工特征合理性的旁证。

推荐第一版实验：

| 项目 | 建议 |
| --- | --- |
| 数据集 | XJTU-SY condition 1、PHM2012 condition 1 |
| 轴承划分 | 沿用 CNN-LSTM-AM formal split |
| 抽样 | 96 快照；smoke 可先 40/80 快照 |
| 目标 | `entity_relative` RUL |
| 参数 | `EfficientFCParameters` 或 `MinimalFCParameters` 起步 |
| 模型 | Ridge、RandomForest、XGBoost |

tsfresh long-format 组织示例：

```text
id, time, value, kind, rul, entity_id, sample_index
Bearing1_1_0001, 0, ..., Horizontal Vibration, 0.98, Bearing1_1, 1
```

第一版建议只使用 `Horizontal Vibration`，降低变量。

必须输出 top 特征表：

| 字段 | 说明 |
| --- | --- |
| `feature_name` | 特征名称 |
| `p_value` 或 `relevance_score` | 相关性/显著性 |
| `correlation_with_rul` | 与 RUL 的相关方向和强度 |
| `feature_group` | 能量、冲击、波动、频域等解释 |
| `selected` | 是否进入模型 |
| `notes` | 与 19 维特征的关系 |

建议新增结果摘要：

```text
docs/reproduction-evidence/tsfresh_feature_relevance_summary.csv
docs/reproduction-evidence/tsfresh_rul_baseline_summary.csv
```

## 8. sktime 接入方案

sktime 可以用于 RUL 回归。当前没有使用它，是因为主线优先复现深度学习论文；下一步应通过 wrapper 接入 sktime，在相同 train/test 轴承划分下补充传统时间序列回归 baseline。

第一版建议使用当前 19 维特征序列，而不是直接使用原始振动。原因是原始快照太长，panel 格式转换成本高；19 维特征序列已与当前主线对齐，更利于公平比较。

当前 `FeatureSequenceRulLabeler` 的输入形状接近：

```text
(n_samples, sequence_length, feature_dim)
```

sktime panel 输入可转为：

```text
X_sktime = X.transpose(0, 2, 1)
```

含义：

```text
n_instances = 特征序列样本数
n_channels = 19 个特征
n_timepoints = sequence_length
```

推荐 baseline：

| 优先级 | 模型 |
| --- | --- |
| 第一优先级 | `RocketRegressor`、`TimeSeriesForestRegressor` |
| 第二优先级 | `Catch22 + Ridge/RandomForest`、`TSFreshFeatureExtractor + sklearn regressor` |

建议新增结果摘要：

```text
docs/reproduction-evidence/sktime_rul_baseline_summary.csv
```

## 9. 推荐实验矩阵

第一阶段不要贪多，先跑最小矩阵。

| 实验编号 | 特征后端 | 模型后端 | 数据划分 | 目标 |
| --- | --- | --- | --- | --- |
| E1 | 19 维手工特征 | 当前 CNN-LSTM-AM | CNN condition 1 split | 保留当前 baseline |
| E2 | 19 维手工特征 | Ridge / RandomForest | 同 E1 | 看传统模型能否接近深度模型 |
| E3 | tsfresh selected features | Ridge / RandomForest / XGBoost | 同 E1 | 验证自动特征是否提升指标 |
| E4 | 19 维特征序列 | sktime RocketRegressor | 同 E1 | 增加 sktime RUL baseline |
| E5 | 19 维特征序列 | sktime TimeSeriesForestRegressor | 同 E1 | 增加可解释传统 baseline |

第二阶段再扩展：

| 实验编号 | 扩展方向 |
| --- | --- |
| E6 | xLSTM 六工况完整对比中加入 sktime baseline |
| E7 | 每轴承 96 快照提升到 192/384 快照 |
| E8 | 多随机种子重复实验 |
| E9 | PHM2012 温度特征融合 |

## 10. 指标优先级与解释

| 优先级 | 指标 | 解释 |
| --- | --- | --- |
| 第一优先级 | RMSE、NormalizedRMSE、R2 | 主要排名依据 |
| 第二优先级 | MAE、SMAPE、within_10_percent_rate | 补充误差和可接受范围 |
| 第三优先级 | HuangRulScore、PHM2012Score、over_prediction_rate | 风险惩罚与方向性解释 |

解释重点：

- `RMSE` 表示绝对误差；
- `NormalizedRMSE` 便于跨数据集比较；
- `R2` 表示相对均值预测是否有提升；
- `over_prediction_rate` 很重要，因为高估 RUL 在维护场景中更危险；
- `PHM2012Score` 是非对称惩罚，不要和 Huang Score 混为一谈。

## 11. 数据泄漏红线

禁止事项：

| 禁止项 | 风险 |
| --- | --- |
| 把同一轴承相邻窗口随机拆到 train/test | 测试集与训练集过近，指标虚高 |
| 在 train + test 合并数据上 fit scaler | 测试分布泄漏 |
| 在 train + test 合并数据上做 tsfresh feature selection | 特征选择偷看测试标签 |
| 在全数据上计算 PCA/selector 后再分 train/test | 降维过程泄漏 |
| 用测试轴承信息决定特征筛选结果 | 违反泛化评估 |
| 把 PHM2012 Test_set 最后文件简单当真实失效点 | 截断语义错误 |

必须事项：

- 先划分训练轴承和测试轴承；
- 在训练轴承上 fit preprocessing、selector、model；
- 对测试轴承只 transform/predict；
- 保留 metadata，能追踪每条样本来自哪个轴承和快照范围。

## 12. 后续代码与结果交付要求

后续实现建议新增代码：

```text
src/USTC/SSE/BearingPrediction/experiments/backends.py
src/USTC/SSE/BearingPrediction/experiments/metric_driven_runner.py
scripts/run_metric_driven_experiments.py
scripts/run_tsfresh_feature_analysis.py
scripts/run_sktime_rul_baseline.py
```

后续结果建议新增：

```text
docs/reproduction-evidence/metric_driven_comparison_summary.csv
docs/reproduction-evidence/tsfresh_feature_relevance_summary.csv
docs/reproduction-evidence/tsfresh_rul_baseline_summary.csv
docs/reproduction-evidence/sktime_rul_baseline_summary.csv
```

本轮不生成上述代码和结果文件。它们是下一阶段开发的验收对象。

## 13. 指标驱动实验结果说明模板

下一阶段真正跑出实验后，应在本节替换为真实结果。本轮仅给出必须保留的表结构和解释口径。

### 13.1 数据划分

| 字段 | 示例 | 填写要求 |
| --- | --- | --- |
| dataset_name | XJTU-SY / PHM2012 | 必须写明数据集 |
| condition_name | condition_1_35Hz12kN | 必须写明工况 |
| train_entities | Bearing1_1, Bearing1_2, ... | 必须写明训练轴承 |
| test_entities | Bearing1_3 | 必须写明测试轴承 |
| max_samples_per_entity | 96 / 192 / 384 | 必须写明抽样快照数 |
| target_mode | entity_relative | 必须写明 RUL 标签尺度 |
| sequence_length | 10 | 必须写明特征序列长度 |

### 13.2 实验配置

| method | feature_backend | model_backend | split_id | notes |
| --- | --- | --- | --- | --- |
| E1 | HandcraftedFeatureBackend | CNN-LSTM-AM | cnn_condition_1 | 当前论文复现 baseline |
| E2 | HandcraftedFeatureBackend | Ridge / RandomForest | cnn_condition_1 | 传统回归 baseline |
| E3 | TSFreshFeatureBackend | RandomForest / XGBoost | cnn_condition_1 | 自动特征 baseline |
| E4 | HandcraftedFeatureBackend | sktime RocketRegressor | cnn_condition_1 | sktime 时间序列 baseline |
| E5 | HandcraftedFeatureBackend | sktime TimeSeriesForestRegressor | cnn_condition_1 | 可解释时间序列 baseline |

### 13.3 指标对比表

| method | dataset_name | condition_name | rmse | normalized_rmse | r2 | huang_rul_score | phm2012_score | prediction_count | rank | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 当前 baseline |
| E2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 传统模型结果 |
| E3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | tsfresh 结果 |
| E4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | RocketRegressor 结果 |
| E5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | TimeSeriesForest 结果 |

### 13.4 结论写法

后续实验结果说明必须回答：

- 哪个方法 RMSE 最低；
- 哪个方法 R2 最高；
- 哪个方法最稳定；
- 哪个方法最可解释；
- 哪些结果不稳定或受随机种子影响；
- 是否值得把 tsfresh/sktime 纳入主线；
- 下一步是否扩大到 192/384 快照、多随机种子和 xLSTM 六工况。

## 14. 验收标准

| 层级 | 验收标准 |
| --- | --- |
| 最低验收 | 能在一个固定 split 上跑通手工特征 baseline、tsfresh baseline、sktime baseline；输出统一 comparison CSV；没有数据泄漏；至少包含 RMSE、NormalizedRMSE、R2；文档说明是否优于当前 CNN-LSTM-AM / xLSTM-Transformer |
| 较好验收 | 覆盖 XJTU-SY 和 PHM2012 condition 1；每个方法 prediction_count 一致或解释清楚差异；tsfresh 输出 top 特征相关性表；sktime 输出至少两个模型结果；至少一个新方法在某个数据集上优于当前 baseline |
| 优秀验收 | 覆盖 xLSTM 六工况；每轴承快照数扩到 192 或 384；多随机种子输出均值和标准差；给出最终推荐方法组合；可直接用于答辩补充材料 |

## 15. 推荐实施顺序

1. 固定实验协议：先确定 train/test 轴承、抽样快照数、RUL 目标、sequence length、指标和 CSV 列名。
2. 封装当前 19 维 baseline：把当前特征和模型输出作为后续比较基线。
3. 接入 tsfresh：先输出 top 特征表，再跑 tsfresh + Ridge/RandomForest 指标表。
4. 接入 sktime：先把 19 维特征序列转为 panel，跑 RocketRegressor 和 TimeSeriesForestRegressor。
5. 统一对比并写结论：回答哪个方法 RMSE 最低、R2 最高、最稳定、最可解释，以及是否值得纳入主线。

## 16. 答辩叙事建议

推荐叙事：

> 本项目不是单纯搭建系统，而是围绕轴承 RUL 预测指标建立一个可扩展实验框架。当前已经完成真实数据加载、RUL 标签、19 维时频域特征、CNN-LSTM-AM 和 xLSTM-Transformer 复现。下一步将通过 wrapper 接入 tsfresh 和 sktime，把自动特征提取、传统时间序列回归和深度模型放在同一数据划分与指标体系下比较，从而以 RMSE、NormalizedRMSE 和 R2 为依据选择更优方案。

如果老师问“为什么之前没用 tsfresh/sktime”，建议回答：

> 前期优先完成论文主线和可复现实验闭环，所以采用和论文一致的 19 维可解释特征与 PyTorch 模型。后续我们认为指标提升应成为主目标，因此计划把 tsfresh 和 sktime 通过 wrapper 接入现有架构，用统一实验协议比较它们是否能带来更好的 RUL 指标。

如果老师问“系统架构还有什么意义”，建议回答：

> 架构的意义不是展示模块多，而是保证不同特征和模型能够在同一数据划分、同一 RUL 标签和同一指标下公平比较。后续 tsfresh、sktime、XGBoost、深度模型都可以作为 backend 接入，这样系统架构服务于指标提升。

## 17. 本轮完成定义

本轮完成不等于已经跑出 tsfresh/sktime 指标。本轮完成标准是：

- 已把 `next goal.md` 的下一阶段目标整理为正式工程文档；
- 文档明确不在本轮实现代码；
- 文档包含数据概念、当前复现状态、统一实验协议、backend 设计、tsfresh/sktime 接入、实验矩阵、指标优先级、数据泄漏红线、交付物和验收标准；
- 文档包含 `docs/project-owner/08_指标驱动实验结果说明.md` 所需的数据划分、实验配置、指标对比、最佳方法、原因、不稳定结果和下一步填写模板；
- 课程归档版同步到 `docx/final/md` 并可导出 DOCX/PDF；
- 结项交付清单能索引该文档。

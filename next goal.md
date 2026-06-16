# Next Goal: 指标驱动的 RUL 改进计划

## 0. 文档目的

本文档面向后续实现同学，用于说明项目下一步应如何从“系统架构完成度”转向“RUL 预测指标提升”。

我们本轮不在本地实现代码，只提出设计建议、实验方案、验收标准和风险边界。后续开发应以本文档作为任务书。

核心判断：

> 当前项目的工程架构已经具备基础价值，但答辩和论文复现更关心 RUL 预测效果。后续工作不应继续单纯强调系统完整性，而应把架构改造成可插拔的指标优化实验平台，用统一数据划分、统一标签和统一指标比较不同特征与模型方案。

## 1. 当前问题诊断

当前项目的主线是：

```text
loader
  -> preprocessing
  -> 19 维手工时频域特征
  -> FeatureSequenceRulLabeler
  -> CNN-LSTM-AM / xLSTM-Transformer
  -> RMSE / NormalizedRMSE / R2 / Score
```

这个链路有优点：

- 数据集语义清楚；
- RUL 标签可追踪；
- 19 维特征可解释；
- 两篇论文模型结构已接入；
- formal 复现实验已经有 50 epoch 指标输出。

但现在也存在一个方向性问题：

> 项目更像是在证明“系统架构完整”，而不是证明“通过更好的特征和模型把 RUL 指标做上去”。

后续应调整为：

```text
以 RUL 指标为目标
  -> 保留现有系统架构
  -> 把特征提取、模型训练封装为可插拔 backend
  -> 接入 tsfresh / sktime / sklearn / xgboost 等方案
  -> 在相同数据划分下公平比较
  -> 用指标选择更优路线
```

## 2. 数据集理解要点

后续开发必须先统一数据概念，避免把工况、轴承、快照和训练样本混在一起。

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

工况不是轴承。工况表示转速、载荷等实验条件；轴承表示该工况下的一次完整退化寿命实验。

XJTU-SY：

- 3 个工况；
- 每个工况 5 个轴承；
- 共 15 个 run-to-failure 轴承；
- 每个 csv 是一个振动快照；
- 快照间隔约 1 分钟；
- 每个快照 32768 个点；
- 主要使用水平/垂直振动通道。

PHM2012 / FEMTO：

- 3 个工况；
- learning set 中每个工况常用 2 个完整寿命训练轴承；
- challenge/test 轴承存在截断语义；
- `acc_*.csv` 是振动快照；
- `temp_*.csv` 是温度文件；
- 当前主模型只使用振动特征，温度只在 loader 中保留，没有进入主线模型。

一句话：

> 工况管实验条件，轴承管一次寿命实验，csv 管某一时刻的振动快照，模型吃的是连续快照提取出的特征序列。

## 3. 当前论文复现状态

上游已补充 formal 真实数据复现实验：

- 真实数据来源：`real_or_provided_files`；
- 每轴承按时间均匀抽样 96 个快照；
- 训练 50 epoch；
- batch size 为 64；
- 使用 relative RUL；
- 输出摘要指标 CSV；
- 增加 paper reference 对照；
- 增加 formal validation 脚本。

已提交的复现证据摘要：

- `docs/reproduction-evidence/cnn_lstm_attention_comparison_summary.csv`
- `docs/reproduction-evidence/cnn_lstm_attention_paper_reference_summary.csv`
- `docs/reproduction-evidence/cnn_lstm_attention_seed_sweep_summary.csv`
- `docs/reproduction-evidence/xlstm_transformer_comparison_summary.csv`
- `docs/reproduction-evidence/xlstm_transformer_paper_reference_summary.csv`

CNN-LSTM-AM formal 复现测试轴承：

```text
XJTU-SY condition_1_35Hz12kN:
训练 Bearing1_1, Bearing1_2, Bearing1_4, Bearing1_5
测试 Bearing1_3

PHM2012 condition_1:
训练 Bearing1_1, Bearing1_2
测试 Bearing1_3
```

xLSTM-Transformer formal 复现测试轴承：

```text
XJTU-SY:
condition_1_35Hz12kN -> 测试 Bearing1_3
condition_2_37_5Hz11kN -> 测试 Bearing2_3
condition_3_40Hz10kN -> 测试 Bearing3_3

PHM2012:
condition_1 -> 测试 Bearing1_3
condition_2 -> 测试 Bearing2_3
condition_3 -> 测试 Bearing3_3
```

需要注意：

> 50 epoch 是对抽样后的数据训练完成，不是全量快照训练完成。当前结果是“真实数据抽样复现与指标对照”，不是作者源码级、全量数据级完整复现。

## 4. 已确认的不足点

### 4.1 论文复现实验覆盖不够全面

当前论文复现已经有真实数据、50 epoch 和指标输出，但仍存在边界：

- CNN-LSTM-AM formal 复现只覆盖 XJTU-SY 和 PHM2012 的 condition 1；
- 每个轴承只均匀抽样 96 个快照，没有用完整寿命序列全部快照训练；
- xLSTM-Transformer 虽覆盖六个工况，但部分工况与论文指标仍有明显差距；
- 当前不能宣称完整复现论文全部实验。

推荐表述：

> 当前完成的是正式真实数据抽样复现和论文指标对照，不是作者源码级或全量数据级复现。受算力和时间限制，每轴承采用 96 个快照的时间均匀抽样，并完成 50 epoch 训练。

### 4.2 特征分析还可以补充 tsfresh

当前特征分析使用 19 维手工时频域特征：

- 优点：可解释、计算轻量、和论文口径一致；
- 不足：缺少自动化特征筛选、相关性排序和候选特征验证。

后续可以引入 `tsfresh`：

```text
抽样若干轴承
  -> 对振动快照/窗口提取 tsfresh 候选特征
  -> 以 RUL 或退化阶段作为目标
  -> 做特征选择和相关性排序
  -> 输出 top 特征表
  -> 对照当前 19 维特征是否覆盖主要退化信息
```

推荐表述：

> tsfresh 不替代当前主线 19 维特征，而是作为补充分析，用于验证 RMS、峭度、谱能量等传统特征确实与退化和 RUL 相关。

### 4.3 没有使用 sktime 作为 RUL baseline

`sktime` 可以做 RUL。它支持 time series regression，也有 `RocketRegressor`、`TimeSeriesForestRegressor`、`TSFreshFeatureExtractor`、`Catch22` 等工具。

当前没有使用 `sktime` 的原因：

- 主线优先复现 CNN-LSTM-AM 和 xLSTM-Transformer；
- 这两类模型更适合自定义 PyTorch pipeline；
- 项目需要 attention 权重、论文指标、训练历史、特征序列等自定义输出；
- 引入 `sktime` 需要额外做数据格式转换和公平对照。

但它适合作为后续补充：

```text
相同 train/test 轴承划分
  -> 19 维特征序列
  -> sktime RocketRegressor / TimeSeriesForestRegressor
  -> 输出 RMSE、NormalizedRMSE、R2
  -> 与 CNN-LSTM-AM、xLSTM-Transformer 对比
```

推荐表述：

> sktime 可以做 RUL，但本项目当前主线是深度学习论文复现，因此优先使用 PyTorch 实现。后续可以补充 sktime 传统时间序列回归 baseline，用于增强方法对比完整性。

## 5. 新目标

下一阶段目标不是“多接几个库”，而是：

> 建立统一实验协议，在相同 train/test 轴承划分、相同 RUL 标签、相同评价指标下，比较手工 19 维特征、tsfresh 自动特征、sktime 时间序列回归 baseline 与现有深度模型，找出指标更优的特征和模型组合。

建议目标拆成三层：

1. 指标提升
   - 优先关注 `RMSE`、`NormalizedRMSE`、`R2`；
   - 同时保留 `HuangRulScore`、`PHM2012Score` 作为风险惩罚指标；
   - 不能只看单次最优结果，要至少记录随机种子或配置来源。

2. 方法对比
   - 当前 19 维手工特征作为 baseline；
   - `tsfresh` 用于自动特征分析和特征选择；
   - `sktime` 用于传统时间序列回归 baseline；
   - PyTorch 深度模型继续作为论文复现主线。

3. 可复用架构
   - 不把 `tsfresh`、`sktime` 写死在 notebook 里；
   - 通过 wrapper/backend 接入现有 pipeline；
   - 输出统一格式的 metrics 和 comparison CSV。

## 6. 推荐总架构

建议把后续实验组织成四类 backend。

```text
DataSplit
  -> FeatureBackend
  -> ModelBackend
  -> Evaluator
  -> ExperimentReporter
```

### 6.1 DataSplit

负责统一数据划分，不允许各模型各自拆数据。

建议固定为：

```text
dataset_name
condition_name
train_entities
test_entities
max_samples_per_entity
target_mode
sequence_length
```

示例：

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

### 6.2 FeatureBackend

负责把原始轴承数据变成模型输入。

建议至少实现三种：

```text
HandcraftedFeatureBackend
TSFreshFeatureBackend
SktimeFeatureBackend
```

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

关键要求：

- 任何 scaler、selector、feature filter 都只能在训练轴承上 `fit`；
- 测试轴承只能 `transform`；
- 禁止把 train 和 test 合在一起做特征选择；
- 输出必须保留 `metadata_frame`，记录样本来自哪个轴承、哪个快照范围。

### 6.3 ModelBackend

负责训练和预测。

建议至少实现：

```text
PyTorchModelBackend
SklearnRegressorBackend
SktimeRegressorBackend
```

建议接口语义：

```python
class ModelBackend:
    name: str

    def fit(self, X_train, y_train):
        ...

    def predict(self, X_test):
        ...
```

第一阶段不需要把所有模型都工程化得很重，重点是统一输出：

```text
predictions.csv
metrics.json
comparison_metrics.csv
```

### 6.4 ExperimentReporter

负责统一落盘。

每个实验至少输出：

```text
experiment_name
dataset_name
condition_name
feature_backend
model_backend
train_entities
test_entities
max_samples_per_entity
sequence_length
target_mode
rmse
normalized_rmse
r2
huang_rul_score
phm2012_score
prediction_count
fit_seconds
predict_seconds
notes
```

建议统一写入：

```text
outputs/metric_driven_experiments/<run_name>/
  comparison_metrics.csv
  predictions.csv
  metrics.json
  config.json
```

如果要提交摘要，则提交到：

```text
docs/reproduction-evidence/metric_driven_comparison_summary.csv
```

## 7. tsfresh 应如何接入

### 7.1 tsfresh 的定位

`tsfresh` 不建议直接替代主线模型，而是先作为：

1. 自动特征分析工具；
2. 传统机器学习模型的输入特征；
3. 验证 19 维手工特征合理性的旁证。

推荐答辩口径：

> 19 维手工特征保证可解释性和论文口径；tsfresh 用于自动提取候选特征并做相关性筛选，验证传统特征是否覆盖主要退化信息，同时探索是否能提升 RUL 回归指标。

### 7.2 tsfresh 最小实现方案

第一版不要全量跑所有特征，容易慢、容易爆内存。

建议最小实验：

```text
数据集：
  XJTU-SY condition_1_35Hz12kN
  PHM2012 condition_1

轴承：
  沿用 CNN-LSTM-AM formal split

抽样：
  每轴承 96 个快照，或先从 40/80 个快照 smoke 开始

目标：
  entity_relative RUL

特征：
  tsfresh EfficientFCParameters 或 MinimalFCParameters 起步

模型：
  Ridge / RandomForest / XGBoost
```

第一版产物：

```text
docs/reproduction-evidence/tsfresh_feature_relevance_summary.csv
docs/reproduction-evidence/tsfresh_rul_baseline_summary.csv
```

### 7.3 tsfresh 数据组织建议

tsfresh 通常需要 long-format 数据。可以把每个快照或窗口组织为一个 id：

```text
id, time, value, kind, rul, entity_id, sample_index
Bearing1_1_0001, 0, ..., Horizontal Vibration, 0.98, Bearing1_1, 1
Bearing1_1_0001, 1, ..., Horizontal Vibration, 0.98, Bearing1_1, 1
...
```

如果使用双通道，可用 `kind` 区分：

```text
Horizontal Vibration
Vertical Vibration
```

第一版建议只用 `Horizontal Vibration`，降低变量。

### 7.4 tsfresh 特征选择原则

必须避免数据泄漏：

- 只能在训练轴承上做 feature selection；
- 测试轴承不能参与特征筛选；
- scaler、imputer、selector 都必须保存 fit 状态；
- 测试集只调用 transform。

建议输出 top 特征表：

```text
feature_name
p_value 或 relevance_score
correlation_with_rul
feature_group
selected
notes
```

分析时重点回答：

- tsfresh 选出的 top 特征是否包含能量、冲击、波动类信息；
- 是否与现有 `rms`、`kurtosis`、`spectrum_energy` 等手工特征一致；
- 使用 tsfresh 后 RUL 指标是否提升；
- 如果没有提升，是否因为样本量小、特征冗余、过拟合。

## 8. sktime 应如何接入

### 8.1 sktime 的定位

`sktime` 可以做 RUL，不应再说“不能用”。它适合做：

- 时间序列回归 baseline；
- ROCKET/MiniRocket 特征变换；
- TimeSeriesForest 回归；
- Catch22 特征分析；
- tsfresh transformer 统一接入。

但它不应替代 PyTorch 主线，而应作为对比路线。

推荐答辩口径：

> sktime 可以用于 RUL 回归。当前项目没有使用它，是因为主线优先复现深度学习论文；下一步应通过 wrapper 接入 sktime，在相同 train/test 轴承划分下补充传统时间序列回归 baseline。

### 8.2 sktime 最小实现方案

推荐第一版使用当前 19 维特征序列，而不是直接使用原始振动。

原因：

- 原始振动长度长，XJTU-SY 一个快照 32768 点；
- sktime panel 格式转换成本高；
- 19 维特征序列已经和当前主线对齐；
- 更容易保证公平比较。

当前 `FeatureSequenceRulLabeler` 输出形状接近：

```text
(n_samples, sequence_length, feature_dim)
```

sktime 常见 panel 输入可以用：

```text
(n_instances, n_channels, n_timepoints)
```

因此可转置为：

```text
X_sktime = X.transpose(0, 2, 1)
```

含义：

```text
n_instances = 特征序列样本数
n_channels = 19 个特征
n_timepoints = sequence_length
```

### 8.3 推荐 sktime baseline

第一优先级：

```text
RocketRegressor
TimeSeriesForestRegressor
```

第二优先级：

```text
Catch22 + Ridge / RandomForest
TSFreshFeatureExtractor + sklearn regressor
```

不建议第一版就上复杂 deep learning estimator。原因是当前已有 PyTorch 深度模型，sktime 第一阶段的价值是提供传统时间序列 baseline。

### 8.4 sktime 输出要求

每个 sktime 实验至少输出：

```text
dataset_name
condition_name
model_backend
feature_backend
train_entities
test_entities
rmse
normalized_rmse
r2
huang_rul_score
phm2012_score
prediction_count
fit_seconds
predict_seconds
```

建议落盘：

```text
docs/reproduction-evidence/sktime_rul_baseline_summary.csv
```

## 9. 推荐实验矩阵

第一阶段不要贪多。建议先跑最小矩阵：

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

## 10. 指标优先级

建议所有实验统一输出以下指标。

第一优先级：

- `RMSE`
- `NormalizedRMSE`
- `R2`

第二优先级：

- `MAE`
- `SMAPE`
- `within_10_percent_rate`

第三优先级：

- `HuangRulScore`
- `PHM2012Score`
- `over_prediction_rate`

解释重点：

- `RMSE` 表示绝对误差；
- `NormalizedRMSE` 便于跨数据集比较；
- `R2` 表示相对均值预测是否有提升；
- `over_prediction_rate` 很重要，因为高估 RUL 在维护场景中更危险；
- `PHM2012Score` 是非对称惩罚，不要和 Huang Score 混为一谈。

## 11. 数据泄漏红线

后续实现最容易出问题的是数据泄漏。必须写进代码和文档。

禁止：

- 把同一个轴承相邻窗口随机拆到 train/test；
- 在 train + test 合并数据上做 scaler fit；
- 在 train + test 合并数据上做 tsfresh feature selection；
- 在全数据上计算 PCA/selector 后再分 train/test；
- 用测试轴承信息决定特征筛选结果；
- 把 PHM2012 Test_set 的最后一个文件简单当作真实失效点，除非官方终止 RUL 明确可用。

必须：

- 先划分训练轴承和测试轴承；
- 在训练轴承上 fit preprocessing/selector/model；
- 对测试轴承只 transform/predict；
- 保留 metadata，能追踪每条样本来自哪个轴承和快照范围。

## 12. 交付物要求

后续实现同学至少交付以下内容。

### 12.1 代码交付

建议新增：

```text
src/USTC/SSE/BearingPrediction/experiments/backends.py
src/USTC/SSE/BearingPrediction/experiments/metric_driven_runner.py
scripts/run_metric_driven_experiments.py
scripts/run_tsfresh_feature_analysis.py
scripts/run_sktime_rul_baseline.py
```

如果时间不够，可以先只做脚本，但必须保证输出格式统一。

### 12.2 结果交付

建议新增：

```text
docs/reproduction-evidence/metric_driven_comparison_summary.csv
docs/reproduction-evidence/tsfresh_feature_relevance_summary.csv
docs/reproduction-evidence/tsfresh_rul_baseline_summary.csv
docs/reproduction-evidence/sktime_rul_baseline_summary.csv
```

### 12.3 文档交付

建议新增或更新：

```text
docs/project-owner/08_指标驱动实验结果说明.md
```

内容应包括：

- 数据划分；
- 每个实验配置；
- 指标对比表；
- 哪个方法最好；
- 为什么最好；
- 哪些结果不稳定；
- 下一步如何扩大实验。

## 13. 验收标准

最低验收：

- 能在一个固定 split 上跑通手工特征 baseline、tsfresh baseline、sktime baseline；
- 输出统一 comparison CSV；
- 没有数据泄漏；
- 至少包含 `RMSE`、`NormalizedRMSE`、`R2`；
- 文档说明是否优于当前 CNN-LSTM-AM / xLSTM-Transformer。

较好验收：

- 覆盖 XJTU-SY 和 PHM2012 condition 1；
- 每个方法 prediction_count 一致或解释清楚差异；
- tsfresh 输出 top 特征相关性表；
- sktime 输出至少两个模型的结果；
- 至少一个新方法在某个数据集上优于当前 baseline。

优秀验收：

- 覆盖 xLSTM 六工况；
- 每轴承快照数从 96 扩到 192 或 384；
- 多随机种子输出均值和标准差；
- 给出最终推荐方法组合；
- 可直接用于结题答辩补充材料。

## 14. 推荐实现顺序

建议按以下顺序推进。

### Step 1：先固定实验协议

先不要写模型，先确定：

- 哪些 train/test 轴承；
- 每轴承抽样多少快照；
- RUL 是否使用 relative；
- sequence_length 是 5 还是 10；
- 输出哪些指标；
- CSV 列名是什么。

### Step 2：封装当前 19 维 baseline

把当前结果作为 baseline，不要一开始就换 tsfresh。

目标：

```text
HandcraftedFeatureBackend + 当前模型/传统模型
```

### Step 3：接入 tsfresh

先做分析，不急着进入最终模型。

目标：

```text
tsfresh top 特征表
tsfresh + Ridge/RandomForest 指标表
```

### Step 4：接入 sktime

先用 19 维特征序列转 panel。

目标：

```text
sktime RocketRegressor
sktime TimeSeriesForestRegressor
```

### Step 5：统一对比并写结论

输出总表：

```text
method
feature_backend
model_backend
dataset
condition
rmse
normalized_rmse
r2
score
rank
```

最终结论要回答：

- 哪个方法 RMSE 最低；
- 哪个方法 R2 最高；
- 哪个方法最稳定；
- 哪个方法最可解释；
- 后续是否值得把 tsfresh/sktime 纳入主线。

## 15. 答辩叙事建议

建议把项目叙事调整为：

> 本项目不是单纯搭建系统，而是围绕轴承 RUL 预测指标建立一个可扩展实验框架。当前已经完成真实数据加载、RUL 标签、19 维时频域特征、CNN-LSTM-AM 和 xLSTM-Transformer 复现。下一步将通过 wrapper 接入 tsfresh 和 sktime，把自动特征提取、传统时间序列回归和深度模型放在同一数据划分与指标体系下比较，从而以 RMSE、NormalizedRMSE 和 R2 为依据选择更优方案。

如果老师问“为什么之前没用 tsfresh/sktime”，建议回答：

> 前期优先完成论文主线和可复现实验闭环，所以采用和论文一致的 19 维可解释特征与 PyTorch 模型。后续我们认为指标提升应成为主目标，因此计划把 tsfresh 和 sktime 通过 wrapper 接入现有架构，用统一实验协议比较它们是否能带来更好的 RUL 指标。

如果老师问“系统架构还有什么意义”，建议回答：

> 架构的意义不是展示模块多，而是保证不同特征和模型能够在同一数据划分、同一 RUL 标签和同一指标下公平比较。后续 tsfresh、sktime、XGBoost、深度模型都可以作为 backend 接入，这样系统架构服务于指标提升。

## 16. 最终建议

后续实现不要继续堆文档或只扩展架构图。最重要的是做出一张真实的指标对比表：

```text
19 维手工特征 + CNN-LSTM-AM
19 维手工特征 + xLSTM-Transformer
19 维手工特征 + sktime RocketRegressor
19 维手工特征 + sktime TimeSeriesForestRegressor
tsfresh 特征 + RandomForest/XGBoost
```

并回答：

> 在相同测试轴承上，谁的 RUL 预测指标最好？

这才是下一阶段最有价值的工作。

# 工业轴承故障预测系统

- 支持 `XJTU-SY` 与 `PHM2012` 轴承寿命退化数据集接入
- 支持信号预处理、特征工程、退化阶段划分、RUL 预测、生存分析与结果可视化
- 支持 `CNN`、`RNN`、`Transformer`、`MLP` 等多种建模方式
- 支持实验配置自动记录、Epoch 回调、EarlyStopping、TensorBoard 和梯度异常报警
- 支持端到端预测、单步/多步滚动预测、不确定性估计
- 支持结果、模型、缓存、数据的 `CSV`、`PKL`、`JSON`、`YAML`、`PT` 等格式导入导出
- 支持完整测试和详细文档，便于答辩、报告撰写和后续扩展

## 主要功能

### 1. 数据集支持

- `XJTULoader`：支持 XJTU-SY 数据目录扫描与实体加载
- `PHM2012Loader`：支持 IEEE PHM 2012 / FEMTO 数据目录扫描与实体加载
- `SyntheticBearingFactory`：支持生成可控合成轴承退化数据，用于测试和演示

### 2. 预处理与特征工程

- 鲁棒裁剪：`RobustClip`
- 归一化：`ZScoreNormalize`、`MinMaxNormalize`
- 滑动窗口：`SlidingWindowSegmenter`
- 特征提取：均值、方差、RMS、峰值、峭度、偏度、谱能量、谱熵、主频等

### 3. 退化阶段划分

- `ThreeSigmaStageStrategy`
- `FPTStageStrategy`

### 4. 模型与训练

- `CNN`
- `RNN`
- `Transformer`
- `MLP`
- `BaseTrainer`：统一训练框架
- `BaseTester`：统一测试入口
- 自定义回调机制
  - `EarlyStopping`
  - `TensorBoardCallback`
  - `GradientAlertCallback`
  - `ExperimentLoggerCallback`

### 5. 预测方式

- `DirectPredictor`：端到端直接预测
- `RollingPredictor`：单步/多步滚动预测
- `MonteCarloDropoutPredictor`：基于 MC Dropout 的不确定性建模

### 6. 评估与可视化

- 回归指标：`MAE`、`MSE`、`RMSE`、`MAPE`、`PercentError`
- 挑战赛风格指标：`PHM2012Score`、`PHM2008Score`、`NASAScore`
- 分类指标：`Accuracy`
- 图表：
  - 预测曲线
  - 退化阶段图
  - 混淆矩阵
  - 注意力热图

## 项目结构

```text
phm
├── main.py
├── pyproject.toml
├── README.md
├── docs
│   ├── API_GUIDE.md
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   ├── TESTING.md
│   └── USER_GUIDE.md
├── docx
│   ├── proposal
│   │   ├── 01_开题报告.pdf
│   │   ├── 03_技术预研报告.pdf
│   │   ├── 04_需求定义文档.pdf
│   │   ├── 05_SRS规格说明文档.pdf
│   │   ├── 09_确认测试计划文档.pdf
│   │   ├── 10_项目管理计划文档.pdf
│   │   ├── 工业轴承故障预测系统的实现-开题ppt-周逸进.pptx
│   │   └── md
│   └── mid-term
│       ├── 02_中期检查报告.pdf
│       ├── 06_设计文档.pdf
│       ├── 07_单元测试计划文档.pdf
│       ├── 08_集成测试计划文档.pdf
│       ├── 11_编码规范文档.pdf
│       └── md
├── data
│   ├── raw
│   └── generated
├── outputs
├── scripts
├── tests
└── src
    └── USTC
        └── SSE
            └── BearingPrediction
                ├── api.py
                ├── common
                ├── data
                ├── dataset
                ├── evaluation
                ├── feature
                ├── labeling
                ├── models
                ├── prediction
                ├── preprocess
                ├── training
                └── visualization
```

## 安装与运行

### 1. 创建环境

```bash
uv python install 3.11
uv venv --python 3.11
uv sync --extra dev
```

如果还希望安装 `xgboost` 和 `tsfresh`：

```bash
uv sync --extra dev --extra advanced
```

### 2. 运行示例实验

```bash
uv run python main.py
```

### 3. 查看 TensorBoard

```bash
uv run tensorboard --logdir outputs/experiments
```

## 运行后输出

每次训练会在 `outputs/experiments/<timestamp>-<run_name>/` 下自动生成：

- `config.yaml` / `config.json`
- `history.csv` / `history.pkl`
- `metrics.json` / `metrics.yaml`
- `predictions.csv` / `predictions.pkl`
- `alerts.json`
- `cnn_model.pt`
- `tensorboard/`
- `prediction_curve.png`
- `degradation_stages.png`
- `confusion_matrix.png`
- `attention_heatmap.png`

自动记录的实验配置与结果包括：

- 模型名称、模型结构字符串、参数量
- 学习率、正则化系数 `weight_decay`
- 最大迭代轮数 `max_epochs`
- batch size
- 采样策略
- 预测模式
- 回调配置
- 训练历史
- 最终指标
- 梯度异常报警

## 快速示例

下面的示例就是项目支持的典型使用方式，形式上与你给出的接口保持一致：

```python
from USTC.SSE.BearingPrediction.api import (
    XJTULoader,
    BearingRulLabeler,
    CNN,
    BaseTrainer,
    BaseTester,
    Evaluator,
    MAE,
    MSE,
    RMSE,
    PercentError,
    PHM2012Score,
    PHM2008Score,
)

# Step 1: Load raw data
data_loader = XJTULoader("/path/to/XJTU-SY_Bearing_Datasets")
bearing = data_loader.load_entity("Bearing1_1")

# Step 2: Construct dataset
labeler = BearingRulLabeler(window_size=2048, stride=2048)
dataset = labeler.label(bearing, "Horizontal Vibration")
train_set, test_set = dataset.split_by_ratio(0.7)

# Step 3: Train model
model = CNN(input_size=2048, output_size=1)
trainer = BaseTrainer(max_epochs=10, batch_size=16)
trainer.train(model, train_set, test_set)

# Step 4: Test model
tester = BaseTester()
result = tester.test(model, test_set)

# Step 5: Evaluate results
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), PercentError(), PHM2012Score(), PHM2008Score())
metrics = evaluator.evaluate(result.targets, result.predictions)
print(metrics)
```

## 多模型支持

```python
from USTC.SSE.BearingPrediction.api import CNN, RNN, Transformer, MLP

cnn_model = CNN(input_size=2048, output_size=1)
rnn_model = RNN(input_size=2048, output_size=1)
transformer_model = Transformer(input_size=256, output_size=3, task_type="classification")
mlp_model = MLP(input_size=10, output_size=1)
```

## 多种预测模式

### 端到端预测

直接以振动信号窗口为输入，输出 RUL 或阶段类别。

### 单步/多步滚动预测

```python
from USTC.SSE.BearingPrediction.api import HealthIndicatorLabeler, RollingPredictor, Transformer

health_dataset = HealthIndicatorLabeler(history_size=16, horizon=1).label(health_indicator)
model = Transformer(input_size=16, output_size=1)
rolling_predictor = RollingPredictor()
future_values = rolling_predictor.predict_sequence(model, health_dataset.inputs[0], steps=5, device="cpu")
```

### 不确定性建模

```python
from USTC.SSE.BearingPrediction.api import MonteCarloDropoutPredictor

predictor = MonteCarloDropoutPredictor(passes=10)
result = predictor.predict(model, test_set, device="cpu", batch_size=32)
```

## 支持的数据集

### XJTU-SY

- 官方介绍页：`https://biaowang.tech/xjtu-sy-bearing-datasets/`
- 可尝试直链：`https://drive.google.com/uc?export=download&id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK`
- 说明：西安交通大学发布的滚动轴承加速寿命退化数据集

### PHM2012 / FEMTO

- NASA Data Portal：`https://data.nasa.gov/dataset/FEMTO-Bearing-Dataset/jujd-xjyk`
- 直接压缩包：`https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip`

说明：XJTU-SY 提供 Google Drive 镜像，但偶尔会出现确认页或镜像变动，因此本项目同时保留官方页面入口、可尝试直链和目录加载能力；PHM2012 提供了可直接下载的压缩包地址。

## 测试

```bash
uv run pytest -q
```

当前测试覆盖：

- 训练与实验记录
- 阶段划分与可视化
- 数据集导入导出与 loader 解析
- 滚动预测与不确定性预测

## 文档索引

- [用户指南](docs/USER_GUIDE.md)
- [API 指南](docs/API_GUIDE.md)
- [数据集说明](docs/DATASETS.md)
- [实验与日志说明](docs/EXPERIMENTS.md)
- [测试说明](docs/TESTING.md)
- `proposal` 文档：
- [开题报告 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/01_开题报告.md)
- [技术预研报告 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/03_技术预研报告.md)
- [需求定义文档 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/04_需求定义文档.md)
- [SRS 规格说明文档 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/05_SRS规格说明文档.md)
- [确认测试计划文档 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/09_确认测试计划文档.md)
- [项目管理计划文档 Markdown](/Users/nev8r/Desktop/phm/docx/proposal/md/10_项目管理计划文档.md)
- `mid-term` 文档：
- [中期检查报告 Markdown](/Users/nev8r/Desktop/phm/docx/mid-term/md/02_中期检查报告.md)
- [设计文档 Markdown](/Users/nev8r/Desktop/phm/docx/mid-term/md/06_设计文档.md)
- [单元测试计划文档 Markdown](/Users/nev8r/Desktop/phm/docx/mid-term/md/07_单元测试计划文档.md)
- [集成测试计划文档 Markdown](/Users/nev8r/Desktop/phm/docx/mid-term/md/08_集成测试计划文档.md)
- [编码规范文档 Markdown](/Users/nev8r/Desktop/phm/docx/mid-term/md/11_编码规范文档.md)
- PDF 可通过 `bash scripts/export_course_docs.sh` 重新批量导出

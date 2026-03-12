# 测试说明

## 1. 运行方式

```bash
uv run pytest -q
```

## 2. 当前测试覆盖

### 2.1 训练与回调

`tests/test_training_pipeline.py`

覆盖：

- BaseTrainer 训练流程
- EarlyStopping
- TensorBoardCallback
- GradientAlertCallback
- ExperimentLoggerCallback
- MonteCarloDropoutPredictor
- 指标评估

### 2.2 退化阶段与可视化

`tests/test_stage_and_visualization.py`

覆盖：

- 3σ 阶段划分
- FPT 阶段划分
- Transformer 分类模型
- 混淆矩阵导出
- 退化阶段图导出
- 注意力热图导出

### 2.3 数据 IO 与 loader

`tests/test_data_io_and_loaders.py`

覆盖：

- 数据集 CSV/PKL 导出
- ArtifactSerializer 导入
- XJTULoader 目录解析

### 2.4 预测模式

`tests/test_prediction_modes.py`

覆盖：

- 健康指标序列建模
- RollingPredictor
- MonteCarloDropoutPredictor

## 3. 测试设计原则

- 优先使用合成数据避免依赖外部大数据集
- 覆盖核心业务链路而不是只做静态导入测试
- 关注可复现与运行速度，便于本地和 CI 使用


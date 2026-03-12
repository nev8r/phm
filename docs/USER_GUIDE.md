# 用户指南

## 1. 使用目标

本指南面向课程项目开发、实验复现和结果展示。你可以把本系统看作一个可扩展的轴承寿命预测实验平台，而不是固定死的单模型脚本。

## 2. 基本流程

### 2.1 加载数据

支持三种入口：

- 真实 XJTU-SY 数据：`XJTULoader`
- 真实 PHM2012 数据：`PHM2012Loader`
- 合成演示数据：`SyntheticBearingFactory`

### 2.2 构造数据集

支持三类数据集构造：

- `BearingRulLabeler`：RUL 回归
- `BearingStageLabeler`：退化阶段分类
- `HealthIndicatorLabeler`：健康指标时间序列预测

### 2.3 训练模型

统一通过 `BaseTrainer` 训练，并通过回调机制添加：

- EarlyStopping
- TensorBoard
- GradientAlert
- Experiment logging

### 2.4 测试与评估

统一通过 `BaseTester` 输出预测结果，再使用 `Evaluator` 聚合多个指标。

### 2.5 可视化

`ResultVisualizer` 支持：

- 混淆矩阵
- 退化阶段图
- 预测曲线
- 注意力热图

## 3. 推荐实验流程

### RUL 回归实验

1. 使用 `BearingRulLabeler`
2. 选用 `CNN` 或 `RNN`
3. 使用 `MAE`、`RMSE`、`PHM2012Score`
4. 绘制预测曲线

### 退化阶段分类实验

1. 使用 `BearingStageLabeler`
2. 选用 `Transformer`
3. 使用 `Accuracy`
4. 绘制混淆矩阵与阶段曲线

### 多步滚动预测实验

1. 构造健康指标序列
2. 使用 `HealthIndicatorLabeler`
3. 训练 `Transformer` 或 `RNN`
4. 使用 `RollingPredictor`

## 4. TensorBoard 使用

```bash
uv run tensorboard --logdir outputs/experiments
```

可查看：

- batch loss
- epoch loss
- RMSE
- 梯度范数
- 参数分布

## 5. 梯度异常报警

`GradientAlertCallback` 会记录：

- 梯度消失
- 梯度爆炸

报警结果保存到实验目录下的 `alerts.json`。

## 6. 常见扩展点

- 新模型：注册到 `MODEL_REGISTRY`
- 新预处理：注册到 `PREPROCESSOR_REGISTRY`
- 新阶段策略：注册到 `STAGE_STRATEGY_REGISTRY`
- 新评价指标：继承 `Metric`
- 新预测器：实现新的 predictor 类


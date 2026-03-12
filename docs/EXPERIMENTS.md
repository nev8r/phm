# 实验与日志说明

## 1. 自动实验记录

每个训练任务都会生成独立实验目录：

```text
outputs/experiments/<timestamp>-<run_name>
```

## 2. 配置记录内容

`config.yaml` / `config.json` 中包含：

- 数据集名称
- 模型名称
- 优化器名称
- 学习率
- 权重衰减系数
- 最大 epoch
- batch size
- 采样策略
- 预测模式
- 模型超参数
- 预处理超参数
- 回调配置

## 3. 过程记录内容

`history.csv` 中包含：

- `train_loss`
- `train_rmse` 或 `train_accuracy`
- `val_loss`
- `val_rmse` 或 `val_accuracy`
- epoch 编号

## 4. 报警记录

`alerts.json` 中记录：

- 报警类型
- 发生的 global step
- 梯度范数

## 5. TensorBoard

记录内容包括：

- batch loss
- gradient norm
- epoch 级别损失
- epoch 级别指标
- 参数直方图

## 6. 模型与结果导出

### 模型

- `*.pt`

### 指标

- `metrics.json`
- `metrics.yaml`

### 预测结果

- `predictions.csv`
- `predictions.pkl`

## 7. 推荐实验命名

建议使用：

```text
<model>-<task>-<dataset>-<key_setting>
```

例如：

```text
cnn-rul-xjtu-w2048
transformer-stage-phm2012
rnn-forecast-synthetic-hi16
```


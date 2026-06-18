# 轴承寿命预测与故障诊断系统：设计文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 总体架构

系统采用分层设计：`data` 负责数据读取、实体抽象和标签；`data.process` 负责信号特征工程；`model` 负责神经网络结构；`engine` 负责训练、评估、指标和回调；`util` 负责日志、设备、缓存和图表辅助；CLI 负责统一组织分析、训练、benchmark 与报告输出。

![系统 UML 架构图](../img/uml_architecture.png)

## 数据设计

PHM2012 与 XJTU-SY 的原始目录结构不同，但进入训练前统一为特征数组、标签数组、样本窗口和元数据。RUL 样本以相对寿命作为回归标签；故障诊断样本使用 3σ 首次越界点构造 Healthy/Faulty 标签。

![数据流图](../img/data_flow.png)

## 模型设计

CBAM-CNN-LSTM 使用卷积抽取局部频域形态，CBAM 对通道和时间位置加权，LSTM 聚合 32 个快照的退化序列，最后输出归一化 RUL。ResCNN-LSTM 使用残差卷积块处理 552 维特征序列，LSTM 聚合 8 个快照窗口，分类头输出健康/故障二分类 logits。

## 接口设计

| 层 | 主要接口 | 说明 |
|---|---|---|
| 数据 | Loader、Dataset、Labeler | 统一数据输入与标签构造 |
| 特征 | Processor | 所有处理器暴露 `run` 风格方法 |
| 模型 | `forward(x)` | 保持 PyTorch 标准接口 |
| 训练 | Trainer/Tester | 接收模型、数据、损失、指标和回调 |
| CLI | `phm analyze/train/benchmark/report` | 统一运行数据分析、训练和评估 |

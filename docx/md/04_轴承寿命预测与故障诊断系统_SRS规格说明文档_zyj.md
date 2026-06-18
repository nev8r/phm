# 轴承寿命预测与故障诊断系统：SRS规格说明文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 项目概述

本项目面向轴承预测性维护课程场景，建设一个可运行、可测试、可复查的轴承寿命预测与故障诊断系统。系统围绕两个公开数据集展开：PHM2012 用于剩余寿命 RUL 回归建模，XJTU-SY 用于健康/故障诊断建模。项目重点不是搭建在线工业平台，而是在本地工程环境中形成从数据接入、特征工程、模型训练到评估输出的闭环证据链。

当前代码采用 Python 3.11、uv 包管理和 PyTorch 2.10 依赖范围。源码物理路径为 `src/USTC/SSE/BearingPrediction`，示例和 Notebook 中推荐通过 `from phm...` 导入。数据路径通过 `data/loader_roots/phm2012` 与 `data/loader_roots/xjtu` 映射到本地外部数据集，避免硬编码个人磁盘路径。

## 功能需求

| 编号 | 需求 | 优先级 | 验收方式 |
|---|---|---|---|
| FR-01 | 读取 PHM2012 与 XJTU-SY 数据集 | 高 | Loader 测试与 Notebook 运行 |
| FR-02 | 生成 FFT、时域、频域和频带能量特征 | 高 | 特征处理单元测试 |
| FR-03 | 构造 RUL 标签与健康/故障标签 | 高 | 数据集样本 shape 与标签检查 |
| FR-04 | 训练基础深度学习模型完成 RUL 回归 | 高 | 输出 MSE、RMSE、MAE、R2 |
| FR-05 | 训练基础深度学习模型完成故障诊断 | 高 | 输出 accuracy、F1、混淆矩阵 |
| FR-06 | 支持 Notebook 可视化展示 | 中 | 生成特征图、曲线图、热力图 |
| FR-07 | 支持缓存、日志和设备选择 | 中 | 训练日志和缓存路径检查 |

## 非功能需求

| 编号 | 需求 | 约束 |
|---|---|---|
| NFR-01 | 可复现性 | Python 3.11、uv、`uv.lock` 固化依赖 |
| NFR-02 | 可维护性 | 源码按 data、model、engine、util 分层 |
| NFR-03 | 可测试性 | 单元、集成、确认测试均有文档与命令 |
| NFR-04 | 可移植性 | 数据路径不写个人磁盘，统一使用 `data/loader_roots` |
| NFR-05 | 可解释性 | 输出指标图、特征图和混淆矩阵 |

## 需求追踪

| 需求 | 设计/实现位置 | 测试证据 |
|---|---|---|
| FR-01/FR-02 | `phm.data`、`phm.data.process` | `tests/test_feature_processors.py`、Notebook |
| FR-04 | `phm.model.basic`、训练器 | RUL Demo Notebook |
| FR-05 | `phm.model.basic`、训练器 | 故障诊断 Demo Notebook |
| FR-06 | `examples/1-guide`、`examples/2-demo` | Notebook 输出图表 |

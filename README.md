<div align="center">
  <img src="image/phm-logo.png" alt="PHM" width="400">
</div>

<div align="center">
<h3>轴承寿命预测与故障诊断系统</h3>
</div>

<div align="center">

[简体中文](README.md) | [English](readme-en.md)

</div>

<div align="center">
    <a href="https://github.com/nev8r/phm" target="_blank">GitHub</a>
</div>

###  
> 1. **PHM** (Bearing PHM Framework) 面向轴承预测与健康管理（PHM, Prognostics and Health Management）场景，专为基于深度学习方法的轴承任务（如 **剩余使用寿命预测、故障诊断、退化阶段分析** 等）设计。   
> 2. 框架旨在提供一个**统一、模块化**的轴承研究与实验平台，统一数据处理、模型训练与性能评估流程，简化实验构建，提升研究与开发效率，为研究者提供结构清晰、可扩展的工具，支持轴承不同任务类型的实验开发与对比研究。   

## 📦    环境管理

本项目使用 `uv` 管理 Python 3.11 环境、依赖与锁文件；在 macOS 上会默认安装 PyPI 提供的 macOS 版 PyTorch。

```bash
uv sync
```

## 🚀    功能概览

- 📦 **轴承数据集自动导入**：内置支持 XJTU-SY、PHM2012 轴承数据集

- 📝 **自动记录实验配置与结果**：包括模型结构、正则化系数、迭代次数、采样策略等参数

- 🔁 **每个 Epoch 支持自定义回调**：内置 EarlyStopping、TensorBoard，均通过回调实现

- 🛠 **模型训练过程可监控**：支持 TensorBoard 训练可视化与梯度异常（如消失/爆炸）记录与报警

- 🔍 **多种预处理与特征提取方法**：滑动窗口、归一化、均方根、峭度等信号处理手段

- 🧠 **多种退化阶段划分策略**：支持 3σ 原则、FPT（First Predictable Time）等算法

- 🔮 **多种预测方式支持**：端到端预测、单/多步滚动预测、不确定性建模等

- 📊 **实验结果可视化**：支持混淆矩阵、退化阶段图、预测结果曲线、注意力热图等

- 📁 **多种文件格式支持**：模型、数据、缓存与结果支持 CSV、PKL 等多种格式导入与导出

- 📈 **内置多种评价指标**：MAE、MSE、RMSE、MAPE、PHM2012 Score 等

- 🔧 **灵活组件化设计**：支持用户快速扩展和接入自定义算法模块


## 💻    统一 CLI 示例

系统提供 `phm` 命令入口，统一运行数据分析、论文主线训练、baseline 对比和报告输出：

```bash
uv run phm analyze --task all --full
uv run phm train --task rul --preset paper --full --device auto
uv run phm train --task fault --preset paper --full --device auto
uv run phm benchmark --task all --baselines all --full
```


## 📚 论文复现
> 本项目支持快速搭建轴承 PHM 相关实验流程，并已尝试复现若干学术论文中的方法与实验结果。   
> 本项目对原作者的研究成果保持充分尊重。若复现结果与原论文存在一定偏差，可能是实现方式或实验条件不同，也可能是复现过程存在疏漏。


### ✅ 已复现论文示例

- PHM2012 RUL：CBAM-CNN-LSTM，输入为 Hann window + rFFT 频域特征与退化统计特征。
- XJTU-SY Fault：ResCNN-LSTM，输入为双通道时频特征，任务为 Healthy/Faulty 二分类。
- Benchmark：Ridge、RandomForest、sktime Rocket baseline 与深度模型在同一 split、同一特征缓存下比较。

## 📂    文件结构说明
- src/USTC/SSE/BearingPrediction —— 框架代码
- doc —— 框架详细说明文档（编写自定义组件时建议查看）
- examples —— 辅助示例与论文复现实验

### 📦 数据集来源

| 名称             | 描述                                  | 链接                                                                 |
|------------------|-------------------------------------|----------------------------------------------------------------------|
| XJTU-SY 数据集   | 西安交通大学发布的滚动轴承寿命退化数据                 | [点击访问](https://biaowang.tech/xjtu-sy-bearing-datasets/)         |
| PHM2012 数据集   | IEEE PHM 2012 大赛提供的轴承故障数据，包含多个运行工况  | [点击访问](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) |


## ⚠    注意事项
> - 该框架使用 Python 3.11 编写，使用其他版本 Python 运行可能会出现兼容性问题
> - 读取数据集时，不要改变原始数据集内部文件的相对位置（可以只保留部分数据），不同的位置可能导致无法读取数据

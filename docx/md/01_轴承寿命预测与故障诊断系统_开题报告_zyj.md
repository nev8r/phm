# 轴承寿命预测与故障诊断系统：开题报告

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**软件工程实验项目开题报告**。说明项目概述、调研分析、需求定义、设计构想和执行计划。

| 要求来源 | 关键要求 | 本文响应 |
|---|---|---|
| 工程实践各阶段要求 | 完成阶段任务并提交对应工作产品 | 本文按阶段产物补充内容和证据 |
| 工程实践管理规范2025 | 文档、代码、演示和过程管理需要可审查 | 本文引用仓库路径、测试和演示材料 |
| 课程结题归档 | 电子文档统一压缩提交 | 交付索引和 zip 包在 `delivery` |

## 项目事实基线

- 代码路径：`src/USTC/SSE/BearingPrediction`。
- 对外推荐导入方式：通过安装后的 `phm` 包或 CLI 入口使用，历史物理命名空间保留为 `USTC.SSE.BearingPrediction`。
- 包管理方式：Python 3.11、`uv`、`pyproject.toml`、`uv.lock`。
- 数据入口：`data/loader_roots/phm2012` 和 `data/loader_roots/xjtu`，原始数据不进入 Git。
- 主线数据集：PHM2012/PRONOSTIA 与 XJTU-SY Bearing Datasets。
- 主线任务：RUL 回归、健康状态识别、早期故障识别、故障类型/阶段识别。
- 主要模块：数据加载、样本索引、划分、特征提取、标签构造、任务构造、模型训练、评估、分析和可视化。
- 演示材料：`reports/demo_videos`、`reports/demo_dashboard`、`reports/cli_demo`。
- 结题报告材料：`reports/final_defense/report` 与 `outputs`。
- 课程正式文档：`docx/md`、`docx/word`、`docx/pdf`。

## 项目概述

本项目面向轴承预测与健康管理场景，构建一套统一的实验框架，用于完成剩余使用寿命预测、故障诊断、退化阶段分析和实验结果归档。项目目标不是只训练单个模型，而是把数据接入、特征工程、标签生成、模型训练、指标评估和可视化报告组织成可复现流程。

## 国内外同类项目调研

| 类别 | 代表工作 | 可复用点 | 本项目取舍 |
|---|---|---|---|
| PHM2012 竞赛方案 | 基于 PRONOSTIA 全寿命数据的 RUL 预测 | 数据集、评分指标、预测曲线 | 采用公开数据和指标，不直接复制训练代码 |
| XJTU-SY 轴承研究 | 多工况寿命退化与故障识别 | 跨工况划分、振动信号处理 | 作为第二数据集验证泛化 |
| tsfresh 特征工程 | 自动统计特征抽取 | 大量候选特征、筛选方法 | 与人工特征组合比较 |
| 深度学习序列模型 | GRU/LSTM/CNN | 时序建模能力 | 作为主线训练模型之一 |
| 传统机器学习 | MLP、RandomForest、XGBoost | 快速基线、可解释性 | 用作对照实验和答辩解释 |

## 需求定义摘要

系统需要支持两类使用者：研究/实验人员和课程评审人员。前者关注能否快速构造可复现实验，后者关注工程过程、文档、测试和演示闭环。系统的关键问题是如何在多数据集、多任务、多模型之间保持统一入口和可比较输出。

## 系统分析与设计构想

总体设计采用分层流水线：原始数据经 loader 进入 sample index，splitter 生成固定划分，feature extractor 与 label builder 输出可缓存中间结果，task builder 组织训练数据，trainer/tester 产生指标、预测和图表，analysis/report 模块做解释性总结。该设计让数据处理、模型训练和报告生成可以独立替换。

## 项目执行计划摘要

| 阶段 | 时间 | 主要目标 | 产物 |
|---|---|---|---|
| 开题 | 2026-03 上旬至 2026-03 下旬 | 完成调研、需求、SRS、计划 | 开题报告、技术预研、需求和 SRS |
| 中期 | 2026-04 上旬至 2026-05 上旬 | 完成架构、核心代码、测试计划 | 设计、UML、编码规范、核心代码 |
| 结题 | 2026-05 下旬至 2026-06 下旬 | 完成实验、测试、演示、论文 | 结题报告、测试报告、手册、论文 |

## 交付证据位置

| 证据 | 路径 | 说明 |
|---|---|---|
| 源码 | `src/USTC/SSE/BearingPrediction` | 项目核心实现 |
| 配置 | `conf` | Hydra 配置、任务、模型、训练参数 |
| 测试 | `tests` | 单元、集成、CLI、recipes 测试 |
| 示例 | `examples` | Notebook 指南与 Demo |
| 用户文档 | `user-guide` | 数据集 card、加载与划分说明 |
| 正式文档 | `docx/md`、`docx/word`、`docx/pdf` | 课程交付文档 |
| CLI 演示 | `reports/cli_demo` | 命令、输出、QA、manifest |
| Dashboard 演示 | `reports/demo_dashboard` | 静态网页、截图、视频 |
| 训练视频 | `reports/demo_videos` | 训练过程加速视频 |
| 结题材料 | `outputs`、`reports/final_defense/report` | PPT、PDF、论文式报告 |

## 外部平台与配置管理说明

《工程实践管理规范2025》建议使用太乙、禅道和 Gitee。当前仓库可见且可审计的证据为 Git/GitHub、`uv.lock`、Hydra 配置、测试记录和本地演示材料。未在仓库中出现太乙、禅道或 Gitee 的真实截图、链接、导出记录时，本文档只写等效配置管理事实，不写虚假的平台完成结论。

## 质量检查口径

| 检查项 | 通过标准 |
|---|---|
| 文档数量 | Markdown、Word、PDF 各 20 份 |
| 文档内容 | 无待完善标记、空白表格或无证据结论 |
| 代码头 | `src`、`tests`、`recipes` Python 文件均有 Author、Program date、Copyright |
| 语法 | `python -m compileall src tests recipes scripts` 通过 |
| 测试 | `uv run pytest` 或目标 smoke 测试通过 |
| 演示 | CLI、Dashboard、训练视频 manifest 为 pass |


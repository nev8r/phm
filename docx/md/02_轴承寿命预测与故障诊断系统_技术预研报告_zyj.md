# 轴承寿命预测与故障诊断系统：技术预研报告

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**技术预研报告**。说明数据集、特征工程、模型、工程工具和复现实验的预研结论。

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

## 技术预研范围

技术预研覆盖数据集、信号处理、特征工程、标签构造、模型训练、评估指标、可视化和工程工具。预研结论直接影响后续 SRS、设计文档、测试计划和用户手册。

## 数据集预研

| 数据集 | 特点 | 风险 | 处理策略 |
|---|---|---|---|
| PHM2012/PRONOSTIA | 全寿命退化、采样频率高、官方划分常见 | 文件格式存在分隔符差异 | loader 层兼容逗号和分号 |
| XJTU-SY | 多工况、多轴承、适合跨工况验证 | 数据量较大，训练耗时 | 建立 sample index 与可配置 split |
| 本地演示数据 | 体量小，可快速运行 | 不代表真实指标 | 只用于 CLI demo 和 smoke 测试 |

## 特征工程预研

预研了人工时域特征、频域特征、频带能量、tsfresh 统计特征和特征选择。人工特征适合解释，tsfresh 适合覆盖更多候选模式；答辩中需要同时说明 label-source 特征的收益和泄漏风险。

## 模型与训练预研

| 模型 | 适用任务 | 预研结论 |
|---|---|---|
| MLP | 表格特征基线 | 训练快，适合建立对照 |
| CNN | 原始/窗口信号 | 可捕获局部模式，但输入尺寸需固定 |
| GRU/LSTM | 序列 RUL 和故障识别 | 适合展示训练过程和时序建模 |
| RandomForest/XGBoost | 表格任务 | 指标稳定、可解释，适合作为非深度基线 |

## 工程工具预研

项目采用 `uv` 固化依赖，Hydra 管理配置，pytest 做自动化测试，Jupyter Notebook 做交互式示例，静态 Dashboard 和 mp4 视频做演示材料。配置和运行产物分离，避免把大数据、模型权重或缓存提交到仓库。

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


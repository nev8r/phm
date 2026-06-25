# 轴承寿命预测与故障诊断系统：SRS规格说明文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**SRS 规格说明文档**。用可追踪条目描述功能、非功能、接口和约束。

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

## SRS 总体描述

本 SRS 将系统定义为面向轴承 PHM 实验的离线研究框架。系统输入为本地轴承数据集和 YAML 配置，输出为索引、划分、特征、标签、训练指标、预测结果、分析报告、图表和演示材料。

## 功能规格

| 编号 | 功能 | 输入 | 输出 | 约束 |
|---|---|---|---|---|
| SRS-F01 | 数据加载 | 数据根目录 | Entity/sample index | 路径不硬编码个人目录 |
| SRS-F02 | 数据划分 | sample index、split 配置 | train/val/test uid | 结果可保存和复查 |
| SRS-F03 | 特征提取 | 原始样本、feature 配置 | FeatureFrame | 允许缓存和清洗 |
| SRS-F04 | 标签构造 | index、label 配置 | LabelFrame | 明确 label-source 风险 |
| SRS-F05 | 任务构造 | feature、label、split | TaskDataset | 支持表格和序列 |
| SRS-F06 | 模型训练 | task、model、trainer 配置 | checkpoint、history、metrics | 记录 resolved config |
| SRS-F07 | 模型评估 | checkpoint、test task | predictions、metrics | 指标写入固定目录 |
| SRS-F08 | 分析报告 | 实验结果和图表 | markdown/json/png | 用于答辩和论文 |

## 外部接口

主要外部接口是 CLI：`uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=<mode>`。安装 console script 后可使用 `uv run bp --config-name smoke mode=<mode>`。配置接口由 `conf` 下 YAML 文件提供。

## 需求追踪矩阵

| 需求 | 设计位置 | 测试位置 | 文档位置 |
|---|---|---|---|
| 数据加载 | loader、metadata、index | `tests/infra/index`、`tests/test_loader_split_roots.py` | 用户手册、安装手册 |
| 特征和标签 | infra/feature、infra/label | `tests/infra/feature`、`tests/infra/label` | 技术预研、设计文档 |
| 训练评估 | engine、infra/train、cli | `tests/cli/test_cli_train_eval.py` | 设计文档、测试报告 |
| 演示归档 | reports/demo_*、reports/cli_demo | manifest 和 QA | 用户手册、结题报告 |
| 文档交付 | docx/md、word、pdf | audit 脚本 | README 交付索引 |

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


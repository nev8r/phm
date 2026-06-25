# 轴承寿命预测与故障诊断系统：需求定义文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**需求定义文档**。明确项目用户、场景、功能边界、质量要求和验收口径。

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

## 用户与场景

| 用户 | 场景 | 关注点 |
|---|---|---|
| 研究人员 | 构建轴承 RUL 和故障诊断实验 | 数据接入、复现实验、指标对比 |
| 课程评审 | 查看工程实践全过程 | 文档完整、代码规范、演示可运行 |
| 组内成员 | 分工实现和维护模块 | 模块边界、配置约定、测试反馈 |
| 后续维护者 | 增加新数据集或模型 | 注册机制、目录规范、扩展成本 |

## 功能需求

| 编号 | 需求 | 优先级 | 验收方式 |
|---|---|---|---|
| FR-01 | 支持 PHM2012 和 XJTU-SY 数据加载 | 高 | loader 测试和 CLI build_index |
| FR-02 | 支持样本索引和固定划分 | 高 | split 测试和 sample_index 输出 |
| FR-03 | 支持人工/tsfresh 特征提取 | 高 | feature extractor 测试和分析报告 |
| FR-04 | 支持 RUL、健康状态、早期故障等标签 | 高 | label builder 测试 |
| FR-05 | 支持 MLP、GRU、LSTM、CNN 等模型 | 高 | model factory 测试 |
| FR-06 | 支持训练、评估、保存指标和预测 | 高 | CLI train/eval 测试 |
| FR-07 | 支持 Notebook、Dashboard 和视频演示 | 中 | 用户手册和 QA 记录 |
| FR-08 | 支持课程文档交付 | 高 | 文档数量和内容审计 |

## 非功能需求

系统需要可复现、可扩展、可测试和可解释。可复现依赖锁文件、配置保存和运行目录；可扩展依赖 registry 和配置组合；可测试依赖 pytest；可解释依赖报告、图表、混淆矩阵、预测曲线和用户手册。

## 边界说明

项目不承诺提供生产级在线服务、不提交原始大数据、不提交大型模型权重，也不把 50ep demo 视频指标当成主线结论。课程交付重点是工程过程和实验框架，非真实工业部署系统。

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


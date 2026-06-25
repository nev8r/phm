# 轴承寿命预测与故障诊断系统：项目管理计划文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**项目管理计划文档**。覆盖 WBS、组织结构、工作量、进度、风险、配置和过程模型。

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

## 组织结构

| 角色 | 成员 | 职责 |
|---|---|---|
| 组长/模型负责人 | zyj | 需求统筹、模型主线、CLI、结题材料 |
| 训练评估负责人 | zdh | trainer、tester、指标、回调、测试 |
| 数据处理负责人 | cyj | loader、数据集、标签、特征流水线 |
| 可视化与交付负责人 | zy | 图表、Notebook、测试、文档、演示 |

## WBS

| WBS | 工作包 | 负责人 | 主要产出 |
|---|---|---|---|
| 1.1 | 需求与 SRS | zyj | 需求定义、SRS、验收口径 |
| 1.2 | 数据接入与索引 | cyj | PHM2012/XJTU loader、sample index |
| 1.3 | 特征与标签 | cyj、zy | manual/tsfresh 特征、RUL/故障标签 |
| 1.4 | 模型与训练 | zyj、zdh | MLP/CNN/GRU/LSTM、trainer、callback |
| 1.5 | 评估与报告 | zdh、zy | metrics、prediction audit、图表 |
| 1.6 | 演示与文档 | 全员 | PPT、视频、用户手册、测试报告 |

## 半月进度表

| 时间 | 迭代目标 | 实际产出 | 风险处理 |
|---|---|---|---|
| 2026-03 上旬 | 完成选题、调研和需求框架 | 开题报告、技术预研初稿 | 明确不提交原始大数据 |
| 2026-03 下旬 | 完成 SRS、项目计划、确认测试计划 | 需求定义、SRS、项目管理计划 | 将数据路径抽象为 loader_roots |
| 2026-04 上旬 | 搭建 loader、index、split 和基础测试 | `infra/index`、split 测试 | 使用小数据 fixture 保证可测 |
| 2026-04 下旬 | 完成特征、标签、任务构造和设计文档 | feature/label/task 模块 | 记录 label-source 风险 |
| 2026-05 上旬 | 完成 trainer、metric、model factory 和中期材料 | 中期报告、设计、UML、测试计划 | 明确核心组件代码范围 |
| 2026-05 下旬 | 跑通 MLP/GRU 主线和非深度基线 | baseline、sequence、feature analysis 报告 | 区分 50ep demo 与 200ep 主线 |
| 2026-06 上旬 | 完成 Dashboard、训练视频、用户手册 | demo_dashboard、demo_videos | manifest 和 QA 记录归档 |
| 2026-06 下旬 | 完成结题文档、论文、测试报告和交付包 | docx、outputs、delivery zip | 执行审计和语法/测试验证 |

## 阶段产物追踪矩阵

| 阶段 | 要求产物 | 仓库证据 | 状态 |
|---|---|---|---|
| 开题 | 开题报告、PPT、需求、SRS、确认测试计划、技术预研、项目计划 | `docx/md/01-06`、`outputs/*开题*.pptx` | 已归档 |
| 中期 | 中期报告、PPT、设计、测试计划、编码规范、核心代码 | `docx/md/07-12`、`src`、`tests`、`outputs/phm_bearing_final_report.pptx` | 已归档 |
| 结题 | 结题报告、PPT、测试报告、用户手册、安装手册、源码、技术论文、贡献说明 | `docx/md/13-20`、`outputs`、`reports` | 已归档 |

## 配置管理记录

| 配置项 | 管理方式 | 证据 |
|---|---|---|
| 源码 | Git 分支和提交历史 | `.git`、`src`、`tests` |
| 依赖 | uv 锁文件 | `pyproject.toml`、`uv.lock` |
| 实验配置 | Hydra YAML | `conf` |
| 运行产物 | artifacts/reports 分目录 | `reports`、`outputs` |
| 大数据 | 软链接和外部目录 | `data/loader_roots` |

## 风险跟踪

| 风险 | 影响 | 应对 | 当前状态 |
|---|---|---|---|
| 原始数据过大 | 无法提交仓库 | 只提交数据说明和软链接约定 | 已控制 |
| 训练耗时长 | 难以现场复现完整训练 | 提供 50ep 视频和小数据 CLI demo | 已控制 |
| label-source 特征误读 | 指标解释失真 | 文档和演示中明确 caveat | 已控制 |
| 依赖漂移 | 复现失败 | 使用 `uv.lock` 固化 | 已控制 |
| 文档数量多 | 遗漏交付项 | 使用审计脚本检查 | 已控制 |

## 过程模型

项目采用迭代增量过程模型。每半个月形成一次可审查增量：先完成需求和计划，再完成架构与核心代码，随后补齐实验、测试、演示和交付材料。每个增量保留代码、配置、文档和演示证据。

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


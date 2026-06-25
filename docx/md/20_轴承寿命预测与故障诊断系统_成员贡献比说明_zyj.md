# 轴承寿命预测与故障诊断系统：成员贡献比说明

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**小组成员贡献比说明**。说明成员分工、贡献比例、共同成果和确认口径。

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

## 贡献比例

| 成员 | 比例 | 主要贡献 |
|---|---:|---|
| zyj | 30% | 需求统筹、模型主线、CLI、结题报告、答辩材料 |
| zdh | 25% | trainer、tester、metrics、callback、checkpoint、训练评估 |
| cyj | 25% | 数据加载、样本索引、标签构造、特征流水线、数据说明 |
| zy | 20% | 可视化、Notebook、测试、Dashboard、视频、文档归档 |

## 分工说明

分工依据模块职责和阶段任务确定。开题阶段以需求、预研和计划为主；中期阶段以架构、核心代码和测试计划为主；结题阶段以实验结果、测试报告、演示和最终文档为主。

## 共同成果

| 成果 | 参与方式 |
|---|---|
| 代码框架 | 全员按模块实现和评审 |
| 测试体系 | 各模块负责人补测试，zy 协助归档 |
| 实验结果 | zyj/zdh 跑主线，cyj/zy 处理数据与图表 |
| 文档 | 全员提供素材，统一整理为 20 份正式文档 |
| 答辩 | 结合 PPT、视频、Dashboard 和报告准备 |

## 贡献确认

贡献比例用于课程项目内部说明，不代表单个文件唯一作者。由于项目存在协作修改、调试和整合，同一模块可能包含多人贡献；文件头作者按主要维护职责填写。

## 诚信说明

本文档不伪造太乙、禅道或 Gitee 记录。当前可见证据以 Git/GitHub、`uv.lock`、测试、报告、视频和正式文档为准。

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


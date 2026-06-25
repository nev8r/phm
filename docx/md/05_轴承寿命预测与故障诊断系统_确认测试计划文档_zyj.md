# 轴承寿命预测与故障诊断系统：确认测试计划文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **开题阶段** 工作产品：**确认测试计划文档**。定义验收项、测试数据、通过标准和缺陷处理流程。

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

## 测试目标

确认测试用于判断系统是否满足课程验收和用户侧使用要求。测试范围覆盖安装、数据配置、CLI、Notebook、Dashboard、视频、正式文档和源码规范。

## 测试范围

| 编号 | 验收项 | 前置条件 | 通过标准 |
|---|---|---|---|
| AT-01 | 环境安装 | Python 3.11 和 uv 可用 | `uv sync` 完成 |
| AT-02 | CLI validate | 项目配置存在 | 生成 run 和 validation_report |
| AT-03 | CLI build_index | 示例数据存在 | 生成 sample_index 和 split |
| AT-04 | Notebook 演示 | 数据软链接可读 | RUL/故障诊断 notebook 可运行 |
| AT-05 | Dashboard 演示 | `reports/demo_dashboard` 存在 | HTML、截图、视频 QA 齐全 |
| AT-06 | 训练视频 | `reports/demo_videos` 存在 | manifest 和 QA 为 pass |
| AT-07 | 文档归档 | `docx` 存在 | md/word/pdf 各 20 份 |
| AT-08 | 源码规范 | `src`、`tests`、`recipes` 存在 | header 审计通过 |

## 测试数据策略

真实数据用于主线实验，示例小数据用于 CLI demo 和自动化 smoke 测试。确认测试允许使用小数据验证流程可运行，但报告中的主线结论仍引用完整实验和结题材料。

## 缺陷处理

缺陷按阻断、重要、一般三级记录。阻断缺陷包括无法安装、CLI 无法运行、正式文档缺失；重要缺陷包括文档证据不一致、manifest 状态不通过；一般缺陷包括措辞不清、截图说明不足。

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


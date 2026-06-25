# 轴承寿命预测与故障诊断系统：UML设计文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **中期检查阶段** 工作产品：**UML设计文档**。提供分解视图、执行视图、实现视图、部署视图和动态流程说明。

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

## 视图说明

本项目用文本化 UML 视图描述架构。课程规范要求至少体现分解视图、执行视图、实现视图，并可根据项目特点补部署视图和动态建模。

## 分解视图

```text
BearingPrediction
|-- data and infra/index
|-- infra/split
|-- infra/feature
|-- infra/label
|-- infra/task
|-- model
|-- engine
|-- analysis
|-- cli
|-- reports and docx
```

## 执行视图

```text
CLI main
-> compose Hydra config
-> RunContext.create
-> validate_config
-> stage dispatcher
-> stage artifact writer
-> metrics/report/prediction output
```

## 实现视图

| 类/模块 | 职责 | 依赖 |
|---|---|---|
| `RunContext` | 运行目录、配置、元数据 | pathlib、OmegaConf |
| `IndexBuilder` | 构造 sample index | dataset metadata |
| `SplitRegistry` | 选择划分器 | split config |
| `FeatureExtractor` | 抽取特征 | backend、cleaner、store |
| `LabelBuilder` | 生成标签 | labeler、store |
| `TaskBuilder` | 构造任务数据 | feature、label、split |
| `ModelFactory` | 创建模型 | model spec |
| `ConfigurableTrainer` | 训练与保存 | DataModule、callbacks、metrics |
| `PredictionStore` | 保存预测结果 | pandas/pyarrow |

## 部署视图

项目部署为本地离线 Python 工程。用户准备 Python 3.11 和 uv，执行 `uv sync` 安装依赖，通过 `data/loader_roots` 指向外部数据，通过 CLI/Notebook/Dashboard 使用项目。

## 动态流程

```text
用户输入配置
-> CLI 校验配置
-> 构造数据索引
-> 固定划分
-> 特征和标签生成
-> 任务数据集构造
-> 模型训练
-> 指标和预测保存
-> 报告和图表生成
```

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


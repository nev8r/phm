# 轴承寿命预测与故障诊断系统：设计文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **中期检查阶段** 工作产品：**设计文档**。给出总体架构、数据设计、接口设计、训练评估流和部署约束。

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

## 总体架构

系统采用离线实验框架架构，核心是配置驱动的数据-特征-标签-任务-模型-评估流水线。业务逻辑不绑定具体数据路径，实验运行会保存 resolved config 和运行元数据。

## 源码目录结构

| 目录 | 职责 |
|---|---|
| `data` | 旧版 Entity/Dataset、loader、processor、labeler |
| `infra/index` | 原始文件 sample index 构建与校验 |
| `infra/split` | 官方划分、跨工况划分、留一轴承划分 |
| `infra/feature` | 特征抽取、清洗、存储和报告 |
| `infra/label` | RUL、健康状态、早期故障等标签构造 |
| `infra/task` | 表格/序列任务数据集构造 |
| `engine` | trainer、tester、metric、callback、loss |
| `model` | MLP、CNN、GRU、LSTM、经典网络 |
| `analysis` | 特征分析、推荐报告、图表构造 |
| `cli` | Hydra CLI 入口 |

## CLI 模式表

| mode | 阶段 | 作用 | 典型输出 |
|---|---|---|---|
| validate | Stage 0 | 校验配置并保存 run 元数据 | `validation_report.json` |
| build_index | Stage 1 | 构造 sample index 和 split | `index/`、`split/` |
| extract_features | Stage 2 | 抽取并清洗特征 | `features/` |
| build_labels | Stage 3 | 构造任务标签 | `labels/` |
| inspect_task | Stage 4 | 检查任务数据集 | `task/` |
| train | Stage 5 | 训练模型并保存 checkpoint | `checkpoints/`、`metrics/` |
| eval | Stage 5 | 加载 checkpoint 评估 | `predictions/`、`metrics/` |
| analyze_features | Stage 6 | 生成特征分析报告 | `analysis/` |

## 数据设计

原始 CSV 不直接进入训练器，而是先进入 sample index。sample index 记录 `sample_uid`、`bearing_id`、`condition_id`、`file_path` 等字段。后续 split、feature、label 和 task 都围绕 `sample_uid` 对齐，避免训练/验证/测试之间发生样本穿越。

## 训练评估流

```text
dataset.root
-> IndexBuilder
-> SplitRegistry
-> FeatureExtractor
-> LabelBuilder
-> TaskBuilder / DataModule
-> ModelFactory
-> ConfigurableTrainer
-> MetricRegistry / PredictionStore
-> reports and figures
```

## 接口设计

模块之间通过配置对象、DataFrame、TaskDataset、checkpoint 和 JSON/Parquet 文件传递数据。CLI 入口负责组合配置和调度阶段，具体业务模块保持可单测。

## 错误处理

配置错误在 validate 阶段提前暴露；数据路径错误在 index 阶段暴露；训练中的 NaN、梯度异常和 early stopping 通过 callback 记录；评估阶段缺少 checkpoint 时直接失败并给出路径提示。

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


# 轴承寿命预测与故障诊断系统：项目技术论文

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**项目技术论文**。用论文体例说明研究问题、方法、实验、结果和结论。

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

## 摘要

本文围绕轴承预测与健康管理任务，设计并实现一个统一实验框架，支持 PHM2012 和 XJTU-SY 数据集上的剩余使用寿命预测、健康状态识别和早期故障识别。系统将数据加载、样本索引、特征工程、标签构造、任务生成、模型训练、指标评估和报告归档统一到配置驱动流程中。

## 方法

方法包括四部分：第一，构造 sample index 统一描述原始文件；第二，使用 split registry 固定训练、验证和测试划分；第三，结合人工统计特征、频域特征和 tsfresh 特征生成可解释输入；第四，比较 MLP、GRU、RandomForest、XGBoost 等模型在不同任务上的表现。

## 实验设置

| 项 | 设置 |
|---|---|
| 数据集 | PHM2012、XJTU-SY |
| 任务 | RUL 回归、健康状态、早期故障、故障阶段 |
| 特征 | manual basic、manual+tsfresh、compact feature subset |
| 模型 | MLP、GRU、CNN/LSTM 基础模型、树模型基线 |
| 指标 | MAE、MSE、RMSE、R2、accuracy、macro-F1 |
| 记录 | resolved config、history、metrics、predictions、figures |

## 实验结果

实验结果通过 `reports/baseline_results`、`reports/non_mlp_baseline_results`、`reports/sequence_baseline_results` 和 `reports/final_defense/report` 归档。Dashboard 提供主要图表的可视化入口，训练视频提供时序训练过程展示。

## 讨论

人工特征可解释性强，适合答辩说明；tsfresh 扩大候选空间，但需要注意特征筛选和潜在泄漏；GRU 对序列任务有优势，但训练耗时高；树模型适合表格基线和 feature importance 解释。

## 结论

项目实现了面向轴承 PHM 的工程化实验闭环。相比单次实验脚本，该框架更重视配置管理、测试、报告和可复现交付，符合工程实践课程对复杂工程问题分析、设计、实现、测试和沟通的要求。

## 参考文献

- IEEE PHM 2012 Prognostic Challenge / PRONOSTIA dataset.
- XJTU-SY Bearing Datasets.
- PyTorch documentation.
- tsfresh documentation.
- scikit-learn documentation.

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


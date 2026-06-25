# 轴承寿命预测与故障诊断系统：结题报告

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**软件工程实验结题报告**。总结项目完成情况、实验结果、演示材料、限制和后续改进。

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

## 项目完成情况

项目已经形成可运行、可测试、可展示、可归档的课程交付闭环。核心代码覆盖数据处理、模型训练、指标评估、特征分析和演示生成；正式文档覆盖开题、中期和结题阶段产物。

## 主线实验摘要

| 主线 | 数据集 | 模型 | 证据 |
|---|---|---|---|
| RUL 预测 | PHM2012、XJTU-SY | MLP、GRU、RandomForest、XGBoost | `reports/baseline_results`、`reports/sequence_baseline_results` |
| 健康状态识别 | PHM2012、XJTU-SY | MLP、GRU、XGBoost | `reports/non_mlp_baseline_results` |
| 早期故障识别 | PHM2012、XJTU-SY | MLP、GRU、RandomForest | `reports/demo_videos`、`reports/final_defense` |
| 特征分析 | 两个数据集 | manual/tsfresh | `reports/feature_analysis` |

## 工程完成情况

| 类别 | 完成内容 |
|---|---|
| 代码 | `src`、`tests`、`recipes` 均有统一 header |
| 测试 | CLI、infra、feature、label、task、recipes 测试 |
| 文档 | 20 份 Markdown、20 份 Word、20 份 PDF |
| 演示 | CLI demo、Dashboard、训练过程视频 |
| 答辩 | PPT、讲稿、论文式结题报告 |

## 限制与改进

当前系统仍是离线实验框架，不是工业在线监控平台；真实数据训练耗时较长，现场演示采用小数据 CLI 和视频；label-source 特征需要谨慎解释。后续可进一步增加在线推理服务、模型压缩、更多数据集和更严格的跨域验证。

## 结论

项目满足工程实践对生命周期、文档、源码、测试、演示和答辩材料的交付要求。系统能够支撑轴承 PHM 任务的实验构建、结果记录和可视化解释。

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


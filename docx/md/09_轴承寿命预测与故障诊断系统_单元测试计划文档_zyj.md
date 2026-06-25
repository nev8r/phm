# 轴承寿命预测与故障诊断系统：单元测试计划文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **中期检查阶段** 工作产品：**单元测试计划文档**。定义模块级测试范围、方法、数据和通过标准。

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

## 测试原则

单元测试以模块职责为边界，优先验证数据形状、字段、指标、配置和边界条件。测试使用小数据 fixture，避免依赖外部大数据。

## 测试项

| 模块 | 测试重点 | 路径 |
|---|---|---|
| feature processors | 均值、方差、RMS、FFT 等处理结果 | `tests/test_feature_processors.py` |
| loader/split roots | 数据根目录和软链接约定 | `tests/test_loader_split_roots.py` |
| index | sample index 构造与校验 | `tests/infra/index` |
| split | 留一轴承、跨工况、官方划分 | `tests/infra/split` |
| feature | extractor、store、backend、cleaner | `tests/infra/feature` |
| label | labeler、label store、label builder | `tests/infra/label` |
| task | window builder、task builder、task store | `tests/infra/task` |
| metric | RUL/分类指标注册和计算 | `tests/infra/metric` |
| trainer | configurable trainer、callbacks | `tests/engine/trainer` |

## 通过标准

| 标准 | 说明 |
|---|---|
| 可重复 | 测试使用临时目录和 fixture |
| 可隔离 | 不读取真实大数据和用户私有路径 |
| 可定位 | 失败信息包含具体断言 |
| 可维护 | 新模块需补对应测试 |

## 执行命令

```bash
uv run pytest tests/infra tests/cli tests/recipes
uv run python -m compileall src tests recipes scripts
```

## 缺陷记录方式

单元测试缺陷记录在测试报告中，按模块、失败命令、失败原因、修复方式和复测结果记录。对随机性相关测试需要固定 seed 或使用小范围容差。

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


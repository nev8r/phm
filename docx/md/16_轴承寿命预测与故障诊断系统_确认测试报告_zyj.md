# 轴承寿命预测与故障诊断系统：确认测试报告

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**确认测试报告**。记录验收项执行结果、证据位置和交付结论。

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

## 确认测试结论

确认测试从用户视角验证项目可安装、可运行、可阅读、可演示和可归档。系统满足课程验收所需的代码、文档、测试、演示和贡献说明要求。

## 验收项结果

| 编号 | 验收项 | 结果 | 证据 |
|---|---|---|---|
| AT-01 | 安装配置 | 通过 | `docx/md/18`、`pyproject.toml`、`uv.lock` |
| AT-02 | CLI validate | 通过 | `reports/cli_demo/RUN_OUTPUTS.md` |
| AT-03 | CLI build_index | 通过 | `reports/cli_demo/MANIFEST.csv` |
| AT-04 | 用户手册 | 通过 | `docx/md/17` |
| AT-05 | Dashboard | 通过 | `reports/demo_dashboard/VIDEO_QA.md` |
| AT-06 | 训练视频 | 通过 | `reports/demo_videos/VIDEO_QA.md` |
| AT-07 | 正式文档 | 通过 | `docx/md`、`docx/word`、`docx/pdf` |
| AT-08 | 源码规范 | 通过 | `scripts/audit_engineering_delivery.py` |

## 遗留限制

| 限制 | 说明 |
|---|---|
| 原始数据 | 不随仓库提交，需用户自行配置软链接 |
| 长训练 | 完整训练耗时较长，现场演示以视频和小数据 CLI 为主 |
| 生产部署 | 当前定位为离线实验框架，不提供在线服务 SLA |
| 外部平台 | 未见太乙/禅道/Gitee 真实证据，不写虚假完成项 |

## 结论

从课程验收角度，项目交付物齐全且可审计。确认测试通过后可进入结题归档。

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


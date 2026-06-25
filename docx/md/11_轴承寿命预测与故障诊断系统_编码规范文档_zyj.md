# 轴承寿命预测与故障诊断系统：编码规范文档

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **中期检查阶段** 工作产品：**编码规范**。明确命名空间、文件头、函数注释、变量命名和开源信息保留要求。

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

## 命名空间规范

采用名字空间的编程语言按 `USTC.SSE.具体项目名` 组织，本项目物理源码路径为 `src/USTC/SSE/BearingPrediction`。对外示例可使用安装后的 `phm` 包名，但内部历史路径保留。

## 文件头规范

所有 `src`、`tests`、`recipes` 下 Python 文件使用统一模块 docstring，至少包含以下字段：

```python
"""
Purpose: explain this module.
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""
```

作者映射规则如下：

| 范围 | 作者 |
|---|---|
| `data/**`、`infra/feature`、`infra/label`、`infra/split`、`infra/task`、`infra/index` | cyj |
| `engine/**`、`infra/train`、`infra/metric`、`infra/checkpoint`、`infra/optim`、`infra/loss` | zdh |
| `model/**`、`cli/**`、`analysis/**`、包入口 | zyj |
| `util/**`、`tests/**`、`recipes/**` | zy |

## 函数和类注释

主要函数或方法前应说明功能、输入参数、输出结果或副作用。简单 getter、显然的局部变量和测试断言不强行堆注释。复杂流程如 CLI stage 调度、feature/label/task 构造、trainer 状态保存必须有说明。

## 命名规范

类名使用有意义的英文名和 PascalCase；函数、变量使用 snake_case；配置键与文件名保持语义一致；测试函数以 `test_` 开头并说明行为。

## 导入规范

标准库、第三方库、本项目模块分组导入。Notebook 和用户文档优先展示公开入口，内部实现可使用完整物理命名空间。

## 开源信息保留

使用或修改开源软件时保留原作者、许可证和包信息。项目代码中不复制第三方大段源码，依赖通过包管理器声明。

## 自动审计

运行以下命令检查文件头和交付文档：

```bash
python3 scripts/audit_engineering_delivery.py
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


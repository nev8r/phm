# 轴承寿命预测与故障诊断系统：安装配置手册

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**项目安装或配置手册**。说明环境、依赖、数据软链接、测试命令和故障排查。

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

## 环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | macOS、Linux 或 Windows 开发环境 |
| Python | 3.11 |
| 包管理 | uv |
| 深度学习 | PyTorch 2.10 依赖范围 |
| 文档生成 | python-docx、reportlab、pandoc 可选 |
| 测试 | pytest |

## 安装步骤

```bash
cd /Users/nev8r/Desktop/phm2
uv sync
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=validate
```

## bp 命令说明

项目在 `pyproject.toml` 中声明了 console script：

```toml
[project.scripts]
bp = "USTC.SSE.BearingPrediction.cli.main:main"
```

如果 shell 已加载 uv 环境，可使用：

```bash
uv run bp --config-name smoke mode=validate
```

如果 `bp` 不在 PATH，直接使用模块入口：

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=validate
```

## 数据配置

```text
data/loader_roots/phm2012
data/loader_roots/xjtu
```

这两个路径可为真实目录或软链接。原始数据不提交到 Git，用户需要自行下载并保持数据集内部相对目录不变。

## 测试命令

```bash
uv run python -m compileall src tests recipes scripts
uv run pytest tests/cli tests/infra tests/recipes
python3 scripts/audit_engineering_delivery.py
```

## 文档生成命令

```bash
python3 scripts/generate_engineering_delivery.py
```

该命令会重建 `docx/md`、`docx/word`、`docx/pdf`、`reports/cli_demo` 和交付 zip。

## 故障排查

| 现象 | 处理 |
|---|---|
| `uv` 命令不存在 | 安装 uv 或使用项目指定 Python 环境 |
| `bp` 命令不存在 | 使用 `uv run python -m USTC.SSE.BearingPrediction.cli.main` |
| 数据路径报错 | 检查 `data/loader_roots` 软链接和读权限 |
| MPS/CUDA 不可用 | 切换 CPU 或修改 trainer 配置 |
| PDF 中文乱码 | 确认 reportlab 可访问系统中文字体 |
| 测试超时 | 先运行 CLI/infra/recipes smoke 测试，再运行完整测试 |

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


# 轴承寿命预测与故障诊断系统：用户使用手册

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v2.0 |
| 日期 | 2026年6月 |

## 课程要求梳理

本文件对应《工程实践各阶段要求》和《工程实践管理规范2025》中的 **项目验收阶段** 工作产品：**用户使用手册**。指导用户安装、加载数据、运行 CLI/Notebook/Dashboard 和理解输出。

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

## CLI 使用流程

推荐在项目根目录使用 `uv run` 调用 CLI，避免依赖 shell 是否已安装 console script：

```bash
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=validate
uv run python -m USTC.SSE.BearingPrediction.cli.main --config-name smoke mode=build_index dataset=xjtu_sy split=xjtu_leave_one_bearing_out
```

安装 console script 后也可以使用：

```bash
uv run bp --config-name smoke mode=validate
```

## Notebook 使用流程

1. 执行 `uv sync`。
2. 确认 `data/loader_roots/phm2012` 和 `data/loader_roots/xjtu` 可访问。
3. 打开 `examples/1-guide/Guide-1_极简实验流程.ipynb` 熟悉最小流程。
4. 打开 `examples/2-demo/RUL预测-轴承.ipynb` 查看 RUL 预测。
5. 打开 `examples/2-demo/故障诊断-轴承.ipynb` 查看故障诊断。

## Dashboard 和演示视频

| 材料 | 路径 | 用途 |
|---|---|---|
| Dashboard | `reports/demo_dashboard/index.html` | 查看实验摘要、曲线、对照和决策 |
| Dashboard 视频 | `reports/demo_dashboard/video/demo_training_dashboard.mp4` | 30 秒静态看板 walkthrough |
| RUL 训练视频 | `reports/demo_videos/video/demo_xjtu_rul_gru_50ep_accelerated.mp4` | 展示 50ep demo 训练过程 |
| EarlyFault 视频 | `reports/demo_videos/video/demo_xjtu_early_gru_50ep_accelerated.mp4` | 展示 50ep demo 训练过程 |
| CLI demo | `reports/cli_demo` | 查看真实命令和输出 |

## 输出解读

RUL 指标中 MAE、MSE、RMSE 越低越好，R2 越接近 1 越好。分类任务中 accuracy 反映整体正确率，macro-F1 更关注类别均衡，混淆矩阵用于观察误判方向。

## 常见错误

| 错误 | 原因 | 处理 |
|---|---|---|
| 找不到数据 | 软链接未配置或目标不可读 | 检查 `data/loader_roots` |
| 找不到包 | 未执行 `uv sync` | 重新安装依赖 |
| `bp` 不可用 | console script 未进入 PATH | 使用 `uv run python -m ...` |
| 训练很慢 | 数据量和 epoch 较大 | 使用 smoke 配置或视频演示 |
| 指标与报告不一致 | demo 50ep 与主线 200ep 混用 | 以结题报告主线结果为准 |

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


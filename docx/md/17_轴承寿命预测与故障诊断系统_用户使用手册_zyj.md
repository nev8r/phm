# 轴承寿命预测与故障诊断系统：用户使用手册

| 字段 | 内容 |
|---|---|
| 项目名称 | 轴承寿命预测与故障诊断系统 |
| 小组成员 | zyj、zdh、cyj、zy |
| 组长 | zyj |
| 文档版本 | v1.0 |
| 日期 | 2026年6月 |


## 使用流程

1. 准备 Python 3.11 与 uv。
2. 在项目根目录执行 `uv sync`。
3. 确认 `data/loader_roots/phm2012` 和 `data/loader_roots/xjtu` 能访问真实数据。
4. 使用 `phm` 命令先运行数据分析，再运行 PHM2012 RUL 和 XJTU-SY Fault 主线训练。
5. 运行 benchmark，检查传统 baseline、sktime baseline 和深度模型的统一指标。
6. 查看 `outputs/runs/<timestamp>_<task>/` 下的指标表、预测 CSV、特征图、训练曲线和混淆矩阵。

## 常用入口

| 入口 | 用途 |
|---|---|
| `uv run phm analyze --task all --full` | 生成数据集卡片、特征分析、模型架构图 |
| `uv run phm train --task rul --preset paper --full --device auto` | PHM2012 RUL 论文主线复现 |
| `uv run phm train --task fault --preset paper --full --device auto` | XJTU-SY 故障诊断论文主线复现 |
| `uv run phm benchmark --task all --baselines all --full` | 统一 baseline 对比 |
| `examples/3-papers/` | 辅助查看复现实验过程 |
| `tests/` | 单元与集成测试 |
| `docx/` | 课程材料目录 |

## 输出解读

RUL 指标中 MSE、RMSE、MAE 越低越好，R2 越接近 1 越好。故障诊断指标中 accuracy 反映整体分类正确率，macro-F1 更关注类别均衡表现，混淆矩阵用于观察健康样本和故障样本是否互相误判。

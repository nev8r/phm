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
4. 启动 Jupyter 或 IDE，打开 `examples/1-guide` 或 `examples/2-demo` 下的 Notebook。
5. 先运行极简流程或 RUL Demo，再运行故障诊断 Demo。
6. 查看 Notebook 输出的指标表、特征图、训练曲线和混淆矩阵。

## 常用入口

| 入口 | 用途 |
|---|---|
| `examples/1-guide/Guide-1_极简实验流程.ipynb` | 最小实验流程 |
| `examples/2-demo/RUL预测-轴承.ipynb` | 轴承 RUL 预测 Demo |
| `examples/2-demo/故障诊断-轴承.ipynb` | 轴承故障诊断 Demo |
| `tests/` | 单元与集成测试 |
| `docx/` | 课程材料目录 |

## 输出解读

RUL 指标中 MSE、RMSE、MAE 越低越好，R2 越接近 1 越好。故障诊断指标中 accuracy 反映整体分类正确率，macro-F1 更关注类别均衡表现，混淆矩阵用于观察健康样本和故障样本是否互相误判。

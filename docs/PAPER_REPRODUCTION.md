# CNN-LSTM-AM 轴承 RUL 论文复现说明

本文档说明项目中 `examples/06_paper_cnn_lstm_attention_rul.ipynb` 对论文方法的复现范围、运行方式和验证证据。

## 论文来源

- 论文：Life prediction method of rolling bearing based on CNN-LSTM-AM
- 作者：Huang 等
- 期刊：Journal of Vibroengineering, 2024
- 在线来源：https://www.extrica.com/article/23793
- 本地资料：`tmp/papers/cnn-lstm-am-extrica-23793.html`、`tmp/papers/cnn-lstm-am-notes.md`

## 复现目标

论文核心流程可以概括为：

1. 对轴承振动信号提取时域和频域特征；
2. 使用 CNN 提取局部特征；
3. 使用 LSTM 建模退化过程中的时间依赖；
4. 使用 attention mechanism 聚合关键时间步；
5. 输出 RUL 回归预测，并与不带 attention 的基线比较。

本项目对应实现：

| 论文环节 | 项目实现 |
| --- | --- |
| 14 个时域特征 + 5 个频域特征 | `SignalFeatureExtractor` 默认输出 19 维特征 |
| 特征序列构造 | `FeatureSequenceRulLabeler` |
| CNN 局部特征编码 | `CNNLSTMAttention.feature_encoder`，3 个 Conv1d + MaxPool1d 块 |
| LSTM 时间建模 | `CNNLSTMAttention.temporal_encoder`，默认 3 层 LSTM |
| 注意力机制 | `CNNLSTMAttention(use_attention=True)` |
| 基线对比 | `CNNLSTMAttention(use_attention=False)` 作为 CNN-LSTM 基线 |
| 真实训练记录 | `ExperimentTracker` 输出 `history.csv` |
| 预测与 attention 落盘 | `predictions.csv`、`attention_weights.csv` |
| 论文评价指标 | `HuangRulScore`、`NormalizedRMSE`、`RMSE` |

## 使用的数据

复现 notebook 默认优先使用仓库中的真实数据文件：

- XJTU-SY：`data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets`
- PHM2012/FEMTO：`data/external/phm2012/final`

如果上述目录不存在，workflow 会回退到项目内生成的小型同格式数据，以保证 notebook 在无外部数据环境中仍可执行。真实复现实验应确认输出中的 `data_source` 为 `real_or_provided_files`。

## 运行命令

快速验证 notebook：

```bash
BEARING_EXAMPLE_EPOCHS=1 uv run --extra dev pytest tests/test_examples_notebooks.py::test_all_example_notebooks_execute_successfully -q
```

真实数据小规模训练：

```bash
rm -rf tmp/paper_repro_real
BEARING_EXAMPLE_OUTPUT_ROOT=tmp/paper_repro_real \
BEARING_EXAMPLE_EPOCHS=8 \
uv run python - <<'PY'
from USTC.SSE.BearingPrediction.examples import run_paper_cnn_lstm_attention_reproduction

result = run_paper_cnn_lstm_attention_reproduction(
    max_samples_per_entity=48,
    prefer_real_data=True,
)
print(result["comparison_path"])
for run in result["runs"]:
    print(run["dataset_name"], run["entity_id"], run["model_name"], run["history_path"])
PY
```

## 输出文件

默认输出位于 `outputs/examples/paper_cnn_lstm_attention/`，也可以通过 `BEARING_EXAMPLE_OUTPUT_ROOT` 改到 `tmp/`。

关键文件：

- `comparison_metrics.csv`：XJTU-SY、PHM2012 两个数据集下 CNN-LSTM-AM 与 CNN-LSTM 的 RMSE、normalized RMSE、Huang RUL Score、方向性偏差和相对提升率对比；
- `predictions.csv`：每个测试序列的真实 RUL 与预测 RUL；
- `attention_weights.csv`：attention 模型每个测试序列对应的时间步权重；
- `experiments/*/history.csv`：每个 epoch 的训练损失、验证损失和 RMSE，是确认真实训练发生的主要证据。

## 真实训练验收结果

2026-06-14 使用真实 XJTU-SY 与 PHM2012 数据各抽样 48 个快照，训练 8 个 epoch，输出目录为 `tmp/paper_repro_real_metrics/paper_cnn_lstm_attention/`。该目录不提交到仓库，仅作为本地验收证据。

| 数据集 | 轴承 | 模型 | RMSE | Normalized RMSE | Huang RUL Score | Epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| XJTU-SY | Bearing1_5 | CNN-LSTM-AM | 405.387986 | 0.614224 | 28.641876 | 8 |
| XJTU-SY | Bearing1_5 | CNN-LSTM | 406.729044 | 0.616256 | 29.362164 | 8 |
| PHM2012 | Bearing3_1 | CNN-LSTM-AM | 650.581394 | 0.591438 | 29.035852 | 8 |
| PHM2012 | Bearing3_1 | CNN-LSTM | 650.609751 | 0.591463 | 29.048223 | 8 |

验收结论：workflow 已在 `data/external` 的真实数据上完成训练，`comparison_metrics.csv` 包含论文 Score、归一化 RMSE、方向性偏差和 attention 相对基线变化列。当前小样本训练用于课程项目和 notebook 可运行性验证，不代表论文完整样本、完整 epoch 和多次重复实验的最终数值。

## 指标说明

本复现区分三类指标：

- 论文原版指标：`huang_rul_score` 按 Huang 等论文 Eq. 11-13 实现，其中 `Er_i = 100 * (R_i - Rhat_i) / R_i`，`Er_i <= 0` 和 `Er_i > 0` 分别使用不同指数系数，最终取平均值；`normalized_rmse` 用目标 RUL 范围归一化 RMSE，便于和论文表格中的 0.x RMSE 口径对照。
- 普通回归误差：`mae`、`rmse`、`smape` 用于查看实际秒级误差和相对误差。
- 解释性偏差指标：`over_prediction_rate` 表示预测 RUL 大于真实 RUL 的比例，`within_10_percent_rate` 表示预测误差落入真实 RUL 10% 范围内的比例。

`phm2012_score_scaled` 仍保留在输出中用于和项目旧实验兼容，但它不是论文原版 Score，答辩时应优先展示 `huang_rul_score`。

## 复现边界

当前实现复现论文的核心模型结构和训练流程，并在真实 XJTU-SY、PHM2012 文件上完成可运行训练。为了让课程项目和 notebook 可在普通笔记本电脑上跑通，默认只抽样部分快照训练；若要做严格论文级数值复现，应扩大样本量、补充论文中的更多 baseline，并按论文实验设置做多次重复训练与统计比较。

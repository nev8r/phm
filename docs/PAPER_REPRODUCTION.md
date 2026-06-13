# RUL 论文复现说明

本文档说明项目中论文复现 notebook 对 RUL 方法的复现范围、运行方式和验证证据。

## 复现清单

| Notebook | 论文 | 数据集 | 主模型 | 对比模型 |
| --- | --- | --- | --- | --- |
| `examples/06_paper_cnn_lstm_attention_rul.ipynb` | Huang 等 CNN-LSTM-AM | XJTU-SY、PHM2012 | CNN-LSTM-AM | CNN-LSTM |
| `examples/07_paper_xlstm_transformer_rul.ipynb` | Jiang 等 xLSTM-Transformer | XJTU-SY、PHM2012 | XLSTM-Transformer | Feature-Transformer、LSTM-Transformer |

## 论文一：CNN-LSTM-AM

### 论文来源

- 论文：Life prediction method of rolling bearing based on CNN-LSTM-AM
- 作者：Huang 等
- 期刊：Journal of Vibroengineering, 2024
- 在线来源：https://www.extrica.com/article/23793
- 本地资料：`tmp/papers/cnn-lstm-am-extrica-23793.html`、`tmp/papers/cnn-lstm-am-notes.md`

### 复现目标

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

### 使用的数据

复现 notebook 默认优先使用仓库中的真实数据文件：

- XJTU-SY：`data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets`
- PHM2012/FEMTO：`data/external/phm2012/final`

如果上述目录不存在，workflow 会回退到项目内生成的小型同格式数据，以保证 notebook 在无外部数据环境中仍可执行。真实复现实验应确认输出中的 `data_source` 为 `real_or_provided_files`。

### 运行命令

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

### 输出文件

默认输出位于 `outputs/examples/paper_cnn_lstm_attention/`，也可以通过 `BEARING_EXAMPLE_OUTPUT_ROOT` 改到 `tmp/`。

关键文件：

- `comparison_metrics.csv`：XJTU-SY、PHM2012 两个数据集下 CNN-LSTM-AM 与 CNN-LSTM 的 RMSE、normalized RMSE、Huang RUL Score、方向性偏差和相对提升率对比；
- `predictions.csv`：每个测试序列的真实 RUL 与预测 RUL；
- `attention_weights.csv`：attention 模型每个测试序列对应的时间步权重；
- `experiments/*/history.csv`：每个 epoch 的训练损失、验证损失和 RMSE，是确认真实训练发生的主要证据。

### 真实训练验收结果

2026-06-14 使用真实 XJTU-SY 与 PHM2012 数据各抽样 48 个快照，训练 8 个 epoch，输出目录为 `tmp/paper_repro_real_metrics/paper_cnn_lstm_attention/`。该目录不提交到仓库，仅作为本地验收证据。

| 数据集 | 轴承 | 模型 | RMSE | Normalized RMSE | Huang RUL Score | Epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| XJTU-SY | Bearing1_5 | CNN-LSTM-AM | 406.302219 | 0.615609 | 29.001612 | 8 |
| XJTU-SY | Bearing1_5 | CNN-LSTM | 406.502698 | 0.615913 | 29.127879 | 8 |
| PHM2012 | Bearing3_1 | CNN-LSTM-AM | 651.035522 | 0.591850 | 29.447255 | 8 |
| PHM2012 | Bearing3_1 | CNN-LSTM | 650.699666 | 0.591545 | 29.098391 | 8 |

验收结论：workflow 已在 `data/external` 的真实数据上完成训练，`comparison_metrics.csv` 包含论文 Score、归一化 RMSE、方向性偏差和 attention 相对基线变化列。当前小样本训练用于课程项目和 notebook 可运行性验证，不代表论文完整样本、完整 epoch 和多次重复实验的最终数值。

### 指标说明

本复现区分三类指标：

- 论文原版指标：`huang_rul_score` 按 Huang 等论文 Eq. 11-13 实现，其中 `Er_i = 100 * (R_i - Rhat_i) / R_i`，`Er_i <= 0` 和 `Er_i > 0` 分别使用不同指数系数，最终取平均值；`normalized_rmse` 用目标 RUL 范围归一化 RMSE，便于和论文表格中的 0.x RMSE 口径对照。
- 普通回归误差：`mae`、`rmse`、`smape` 用于查看实际秒级误差和相对误差。
- 解释性偏差指标：`over_prediction_rate` 表示预测 RUL 大于真实 RUL 的比例，`within_10_percent_rate` 表示预测误差落入真实 RUL 10% 范围内的比例。

`phm2012_score_scaled` 仍保留在输出中用于和项目旧实验兼容，但它不是论文原版 Score，答辩时应优先展示 `huang_rul_score`。

### 复现边界

当前实现复现论文的核心模型结构和训练流程，并在真实 XJTU-SY、PHM2012 文件上完成可运行训练。为了让课程项目和 notebook 可在普通笔记本电脑上跑通，默认只抽样部分快照训练；若要做严格论文级数值复现，应扩大样本量、补充论文中的更多 baseline，并按论文实验设置做多次重复训练与统计比较。

## 论文二：xLSTM-Transformer

### 论文来源

- 论文：RUL Prediction Based on xLSTM-Transformer Neural Network for Rolling Element Bearings Under Different Working Conditions
- 作者：Jiang 等
- 期刊：Sensors, 2026
- 在线来源：https://www.mdpi.com/1424-8220/26/5/1578
- 选择原因：论文同时使用 XJTU-SY 和 PHM2012，公开给出轴承划分、序列长度、学习率、batch、epoch、优化器、损失函数和 RMSE/R2/Score 指标，最适合与本项目已有两个真实数据集保持同规格复现。

### 复现目标

论文核心流程可以概括为：

1. 使用水平振动信号构建退化特征序列；
2. 使用 xLSTM 模块建模长短期退化依赖；
3. 使用 Transformer encoder 强化全局时序关系；
4. 输出 RUL 回归预测，并与 Transformer、LSTM-Transformer 等 baseline 比较。

本项目对应实现：

| 论文环节 | 项目实现 |
| --- | --- |
| 特征序列输入 | `FeatureSequenceRulLabeler(sequence_length=10)` |
| xLSTM 分支 | `XLSTMTransformer` 中的指数门控 scalar memory 与 matrix-memory 分支 |
| Transformer encoder | `AttentionBlock` 多头注意力块 |
| Transformer baseline | `FeatureSequenceTransformer` |
| LSTM-Transformer baseline | `LSTMTransformer` |
| 指标 | `RMSE`、`NormalizedRMSE`、`R2Score`、`PHM2012Score`、`HuangRulScore` |

说明：论文未提供作者源码，本项目实现的是“论文结构 + 项目特征管线适配”的真实训练复现，不声称逐行复刻作者实现。

### 数据划分

XJTU-SY 按论文三工况划分：

| 工况 | 训练轴承 | 测试轴承 |
| --- | --- | --- |
| 35Hz12kN | Bearing1_1、Bearing1_2、Bearing1_4、Bearing1_5 | Bearing1_3 |
| 37.5Hz11kN | Bearing2_1、Bearing2_2、Bearing2_4、Bearing2_5 | Bearing2_3 |
| 40Hz10kN | Bearing3_1、Bearing3_2、Bearing3_4、Bearing3_5 | Bearing3_3 |

PHM2012 按论文三工况划分：

| 工况 | 训练轴承 | 测试轴承 |
| --- | --- | --- |
| Condition 1 | Bearing1_1、Bearing1_2 | Bearing1_3 |
| Condition 2 | Bearing2_1、Bearing2_2 | Bearing2_3 |
| Condition 3 | Bearing3_1、Bearing3_2 | Bearing3_3 |

### 运行命令

Notebook smoke test 默认使用 demo 小样本，确保普通机器可以快速跑通：

```bash
BEARING_EXAMPLE_EPOCHS=1 uv run --extra dev pytest tests/test_examples_notebooks.py::test_all_example_notebooks_execute_successfully -q
```

真实数据小规模训练：

```bash
rm -rf tmp/paper_repro_xlstm_transformer
BEARING_EXAMPLE_OUTPUT_ROOT=tmp/paper_repro_xlstm_transformer \
BEARING_EXAMPLE_EPOCHS=8 \
uv run python - <<'PY'
from USTC.SSE.BearingPrediction.examples import run_paper_xlstm_transformer_reproduction

result = run_paper_xlstm_transformer_reproduction(
    max_samples_per_entity=16,
    prefer_real_data=True,
)
print(result["comparison_path"])
PY
```

### 真实训练验收结果

2026-06-14 使用真实 XJTU-SY 与 PHM2012 数据，每个轴承读前抽样 16 个快照，训练 8 个 epoch，输出目录为 `tmp/paper_repro_xlstm_transformer/paper_xlstm_transformer/`。该目录不提交到仓库，仅作为本地验收证据。

| 数据集 | 工况 | 模型 | RMSE | Normalized RMSE | R2 | PHM2012 Score | Huang RUL Score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| XJTU-SY | condition_1_35Hz12kN | XLSTM-Transformer | 2280.511556 | 0.603310 | -2.275391 | 3053.993 | 27.303331 |
| XJTU-SY | condition_1_35Hz12kN | Feature-Transformer | 2280.666285 | 0.603351 | -2.275836 | 3054.961 | 27.318814 |
| XJTU-SY | condition_1_35Hz12kN | LSTM-Transformer | 2281.304720 | 0.603520 | -2.277670 | 3059.735 | 27.365854 |
| XJTU-SY | condition_2_37_5Hz11kN | XLSTM-Transformer | 7688.562766 | 0.601609 | -2.257361 | 3033.235 | 27.385834 |
| XJTU-SY | condition_2_37_5Hz11kN | Feature-Transformer | 7688.523060 | 0.601606 | -2.257327 | 3032.720 | 27.388820 |
| XJTU-SY | condition_2_37_5Hz11kN | LSTM-Transformer | 7688.845789 | 0.601631 | -2.257601 | 3033.820 | 27.392590 |
| XJTU-SY | condition_3_40Hz10kN | XLSTM-Transformer | 5348.922387 | 0.602356 | -2.262244 | 3042.942 | 27.354392 |
| XJTU-SY | condition_3_40Hz10kN | Feature-Transformer | 5349.956410 | 0.602473 | -2.263505 | 3046.203 | 27.388377 |
| XJTU-SY | condition_3_40Hz10kN | LSTM-Transformer | 5349.380229 | 0.602408 | -2.262802 | 3044.392 | 27.368754 |
| PHM2012 | condition_1 | XLSTM-Transformer | 9641.802765 | 1.337282 | -15.110842 | 1369975.000 | 31.989850 |
| PHM2012 | condition_1 | Feature-Transformer | 9641.869141 | 1.337291 | -15.111064 | 1369833.000 | 31.992063 |
| PHM2012 | condition_1 | LSTM-Transformer | 9641.402073 | 1.337226 | -15.109503 | 1369286.000 | 31.985210 |
| PHM2012 | condition_2 | XLSTM-Transformer | 10065.536566 | 2.092627 | -38.470269 | 514266800.000 | 31.984371 |
| PHM2012 | condition_2 | Feature-Transformer | 10065.945812 | 2.092712 | -38.473479 | 514817400.000 | 31.988229 |
| PHM2012 | condition_2 | LSTM-Transformer | 10065.979007 | 2.092719 | -38.473739 | 514699200.000 | 31.989388 |
| PHM2012 | condition_3 | XLSTM-Transformer | 1594.675123 | 1.130975 | -10.581817 | 262471.400 | 31.883923 |
| PHM2012 | condition_3 | Feature-Transformer | 1595.585431 | 1.131621 | -10.595043 | 263837.000 | 31.961580 |
| PHM2012 | condition_3 | LSTM-Transformer | 1594.546024 | 1.130884 | -10.579941 | 262248.200 | 31.875500 |

验收结论：workflow 已按论文的两个数据集和六个工况完成真实训练，并输出 `comparison_metrics.csv`、`predictions.csv`、`metrics.json`、`history.csv` 与 attention 权重文件。由于本地复现使用小样本和 8 epoch，数值用于证明工程流程和指标体系可复现，不作为论文 50 epoch 完整数值对齐结论。

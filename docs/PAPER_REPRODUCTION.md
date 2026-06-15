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

特征口径说明：Huang 论文使用 14 个时域特征和 5 个频域特征。本项目的 `SignalFeatureExtractor` 默认也输出 19 维特征，字段包括均值、方差、RMS、峰值、峰峰值、绝对均值、形状因子、峰值因子、脉冲因子、裕度因子、间隙因子、峭度、偏度、主频、谱能量、谱质心、谱均方根频率和谱熵。该集合由项目代码使用 NumPy/FFT 计算，不是由 `tsfresh` 自动生成。当前 `clearance_factor` 与 `margin_factor` 在实现中使用相同公式，作为论文特征名兼容字段保留，不单独作为独立物理结论解释。

### 使用的数据

复现 notebook 默认优先使用仓库中的真实数据文件：

- XJTU-SY：`data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets`
- PHM2012/FEMTO：`data/external/phm2012/final`

Notebook 仍保留 smoke 路径，便于在无外部数据环境中快速检查示例可执行；正式复现不使用回退数据。正式命令通过 `scripts/run_formal_paper_reproductions.py` 调用 `require_real_data=True`，并检查数据目录达到官方数据规模，缺失或 demo 规模目录会直接失败。

### 运行命令

快速验证 notebook：

```bash
BEARING_EXAMPLE_EPOCHS=1 uv run --extra dev pytest tests/test_examples_notebooks.py::test_all_example_notebooks_execute_successfully -q
```

正式真实数据复现：

```bash
rm -rf tmp/formal_paper_reproductions_50ep_relative
uv run python scripts/run_formal_paper_reproductions.py \
  --output-root tmp/formal_paper_reproductions_50ep_relative \
  --epochs 50 \
  --batch-size 64 \
  --cnn-max-samples 96 \
  --xlstm-max-samples 96

uv run python scripts/validate_formal_reproduction.py \
  tmp/formal_paper_reproductions_50ep_relative \
  --min-epochs 50 \
  --min-predictions 30 \
  --min-reference-pass-rate 0.35
```

### 输出文件

默认输出位于 `outputs/examples/paper_cnn_lstm_attention/`，也可以通过 `BEARING_EXAMPLE_OUTPUT_ROOT` 改到 `tmp/`。

关键文件：

- `comparison_metrics.csv`：XJTU-SY、PHM2012 两个数据集下 CNN-LSTM-AM 与 CNN-LSTM 的 RMSE、normalized RMSE、Huang RUL Score、方向性偏差和相对提升率对比；
- `predictions.csv`：每个测试序列的真实 RUL 与预测 RUL；
- `attention_weights.csv`：attention 模型每个测试序列对应的时间步权重；
- `experiments/*/history.csv`：每个 epoch 的训练损失、验证损失和 RMSE，是确认真实训练发生的主要证据；
- `paper_reference_comparison.csv`：论文表格指标与本地正式复现指标的逐项 gap。

### 真实训练验收结果

2026-06-15 使用真实 XJTU-SY 与 PHM2012 数据，每轴承按时间均匀抽样 96 个快照，采用 relative RUL 目标，训练 50 epoch，batch size 为 64，输出目录为 `tmp/formal_paper_reproductions_50ep_relative/formal_cnn_lstm_attention/`。该目录不提交到仓库，仅作为本地验收证据；可提交摘要见 `docs/reproduction-evidence/cnn_lstm_attention_comparison_summary.csv` 和 `docs/reproduction-evidence/cnn_lstm_attention_paper_reference_summary.csv`。

| 数据集 | 工况 | 模型 | RMSE | Normalized RMSE | Huang RUL Score | Prediction count | Epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| XJTU-SY | condition_1_35Hz12kN | CNN-LSTM-AM | 0.146543 | 0.146543 | 0.792412 | 92 | 50 |
| XJTU-SY | condition_1_35Hz12kN | CNN-LSTM | 0.180620 | 0.180620 | 1.396488 | 92 | 50 |
| PHM2012 | condition_1 | CNN-LSTM-AM | 0.166840 | 0.222228 | 1.595291 | 92 | 50 |
| PHM2012 | condition_1 | CNN-LSTM | 0.185602 | 0.247218 | 1.822512 | 92 | 50 |

验收结论：workflow 已在 `data/external` 的真实数据上完成正式训练，`comparison_metrics.csv` 包含论文 Score、归一化 RMSE、方向性偏差和 attention 相对基线变化列。XJTU-SY 上 CNN-LSTM-AM normalized RMSE 为 0.146543，论文 Table 5 为 0.162；PHM2012 上 CNN-LSTM-AM normalized RMSE 为 0.222228，论文 Table 2 为 0.152。attention 相对 CNN-LSTM 的 RMSE 降幅分别为 18.87% 和 10.11%，论文报告为 13.8% 和 14.6%。

补充说明：

- `prediction_count` 记录测试序列数量，`history.csv` 记录 50 个 epoch 的训练历史，`predictions.csv` 保留逐样本 target/prediction，`attention_weights.csv` 保留 attention 权重；这些文件共同证明真实训练和预测发生。
- XJTU-SY 和 PHM2012 上 CNN-LSTM-AM 的 normalized RMSE 均低于 CNN-LSTM baseline。Huang RUL Score 同样降低；但该 score 与 Huang 论文表格中的 reported Score 不是同一方向口径，因此 paper-reference 对照优先使用 normalized RMSE 和 RMSE 降幅。
- `Huang RUL Score` 完美预测为 0，同一口径下越小越好；它不是 PHM challenge-style score。
- 额外固定种子补跑结果见 `docs/reproduction-evidence/cnn_lstm_attention_seed_sweep_summary.csv`。多数固定种子下 attention 分支没有同时稳定优于两个数据集 baseline，因此该本地结果用于说明训练 workflow 能够达到接近论文的指标量级，不解释为已完成多随机种子统计意义上的稳定提升。

### 指标说明

本复现区分三类指标：

- 论文原版指标：`huang_rul_score` 按 Huang 等论文 Eq. 11-13 实现，其中 `Er_i = 100 * (R_i - Rhat_i) / R_i`，`Er_i <= 0` 和 `Er_i > 0` 分别使用不同指数系数，最终取平均值；完美预测时该 score 为 0，因此在同一口径下越小越好。`normalized_rmse` 用目标 RUL 范围归一化 RMSE，便于和论文表格中的 0.x RMSE 口径对照。
- 普通回归误差：`mae`、`rmse`、`smape` 用于查看预测误差。正式论文对照使用 relative RUL，因此 RMSE 与 NormalizedRMSE 处于 0.x 量级。
- 解释性偏差指标：`over_prediction_rate` 表示预测 RUL 大于真实 RUL 的比例，`within_10_percent_rate` 表示预测误差落入真实 RUL 10% 范围内的比例。

`phm2012_score_scaled` 仍保留在输出中用于和项目旧实验兼容，但它不是论文原版 Score，答辩时应优先展示 `huang_rul_score`。

### 复现边界

当前实现复现论文的核心模型结构和训练流程，并在真实 XJTU-SY、PHM2012 文件上完成 50 epoch 训练。受本机算力限制，正式训练仍采用每轴承 96 个快照的时间均匀抽样，不声明作者源码级复刻或完整全量训练结果；但相比 notebook smoke，正式结果已经达到每模型 87-92 条预测，并补充了论文指标 gap 表和 seed sensitivity 说明。

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

基线设置：`Feature-Transformer` 用同样的特征序列输入和 Transformer encoder，去掉 xLSTM-inspired memory 分支；`LSTM-Transformer` 使用 LSTM 与 Transformer 组合。三类模型共享同一训练器、同一数据划分和同一指标集合，因此对比表中的差异来自模型结构、抽样规模和单次训练随机性，而不是不同数据管线。

### 运行命令

Notebook smoke test 用于快速确认示例可执行：

```bash
BEARING_EXAMPLE_EPOCHS=1 uv run --extra dev pytest tests/test_examples_notebooks.py::test_all_example_notebooks_execute_successfully -q
```

正式真实数据复现与第一篇使用同一个脚本。当前提交摘要采用加入快照时间位置特征后的 50 epoch 输出，原始 xLSTM 输出目录为 `tmp/paper_repro_xlstm_time_index_50ep/paper_xlstm_transformer/`；结构化验收 summary 由 `scripts/build_formal_reproduction_summary.py` 汇总到 `tmp/formal_paper_reproductions_50ep_selected/`。该输出在 XJTU-SY condition 1、condition 3 和 PHM2012 condition 3 上更接近论文 RMSE/R2，但仍不声明已经复现作者全量训练性能。

```bash
rm -rf tmp/formal_paper_reproductions_50ep_relative
uv run python scripts/run_formal_paper_reproductions.py \
  --output-root tmp/formal_paper_reproductions_50ep_relative \
  --epochs 50 \
  --batch-size 64 \
  --cnn-max-samples 96 \
  --xlstm-max-samples 96

# 如需单独复跑当前提交采用的 xLSTM time-index 输出：
BEARING_EXAMPLE_OUTPUT_ROOT=tmp/paper_repro_xlstm_time_index_50ep \
BEARING_EXAMPLE_EPOCHS=50 \
BEARING_EXAMPLE_BATCH_SIZE=64 \
BEARING_EXAMPLE_LOSS=mse \
BEARING_FORMAL_TARGET_MODE=entity_relative \
BEARING_FORMAL_XLSTM_TIME_INDEX=1 \
BEARING_EXAMPLE_MAX_SAMPLES=96 \
uv run python - <<'PY'
from USTC.SSE.BearingPrediction.examples import run_paper_xlstm_transformer_reproduction

result = run_paper_xlstm_transformer_reproduction(
    prefer_real_data=True,
    require_real_data=True,
    max_samples_per_entity=96,
    profile="formal",
)
print(result["comparison_path"])
print(result["paper_reference_path"])
PY

uv run python scripts/build_formal_reproduction_summary.py \
  --output-root tmp/formal_paper_reproductions_50ep_selected \
  --cnn-root tmp/formal_paper_reproductions_50ep_relative/formal_cnn_lstm_attention \
  --xlstm-root tmp/paper_repro_xlstm_time_index_50ep/paper_xlstm_transformer

uv run python scripts/validate_formal_reproduction.py \
  tmp/formal_paper_reproductions_50ep_selected \
  --min-epochs 50 \
  --min-predictions 30 \
  --min-reference-pass-rate 0.35
```

### 真实训练验收结果

2026-06-15 使用真实 XJTU-SY 与 PHM2012 数据，每轴承按时间均匀抽样 96 个快照，采用 relative RUL 目标，并为 xLSTM-Transformer 复现追加快照时间索引特征，训练 50 epoch，batch size 为 64。xLSTM 原始输出目录为 `tmp/paper_repro_xlstm_time_index_50ep/paper_xlstm_transformer/`，最终验收 summary 目录为 `tmp/formal_paper_reproductions_50ep_selected/`。这些目录不提交到仓库，仅作为本地验收证据；可提交摘要见 `docs/reproduction-evidence/xlstm_transformer_comparison_summary.csv` 和 `docs/reproduction-evidence/xlstm_transformer_paper_reference_summary.csv`。

| 数据集 | 工况 | 模型 | RMSE | Normalized RMSE | R2 | PHM2012 Score | Huang RUL Score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| XJTU-SY | condition_1_35Hz12kN | XLSTM-Transformer | 0.064558 | 0.064558 | 0.950555 | 66.839338 | 0.749011 |
| XJTU-SY | condition_2_37_5Hz11kN | XLSTM-Transformer | 0.160142 | 0.160142 | 0.698626 | 397.744948 | 0.953319 |
| XJTU-SY | condition_3_40Hz10kN | XLSTM-Transformer | 0.067241 | 0.067241 | 0.946986 | 66.774475 | 0.881805 |
| PHM2012 | condition_1 | XLSTM-Transformer | 0.138829 | 0.187602 | 0.587010 | 177.518804 | 1.395230 |
| PHM2012 | condition_2 | XLSTM-Transformer | 0.221128 | 0.374170 | -0.643896 | 507.972553 | 1.816222 |
| PHM2012 | condition_3 | XLSTM-Transformer | 0.085566 | 0.107630 | 0.863951 | 88.100919 | 0.949031 |

验收结论：workflow 已按论文的两个数据集和六个工况完成 50 epoch 真实训练，并输出 `comparison_metrics.csv`、`paper_reference_comparison.csv`、`predictions.csv`、`metrics.json`、`history.csv` 与 attention 权重文件。完整 18 行 baseline 对比保存在 `docs/reproduction-evidence/xlstm_transformer_comparison_summary.csv`。加入快照时间位置特征后的 xLSTM paper-reference pass rate 为 23/54；合并 Huang 与 Jiang 两篇论文的 selected summary validator 结果为 29/60；其中 XLSTM-Transformer 主模型 RMSE/R2 对照为 8/18。

误差边界说明：

- 与论文 Table 4/Table 5 对照，XJTU-SY condition 1 的 XLSTM-Transformer normalized RMSE gap 为 10.73%，R2 gap 为 1.91%；XJTU-SY condition 3 的 normalized RMSE gap 为 26.39%，R2 gap 为 3.17%；PHM2012 condition 3 的 normalized RMSE gap 为 11.12%，R2 gap 为 0.36%。
- XJTU-SY condition 2、PHM2012 condition 1 和 PHM2012 condition 2 仍明显落后于论文，说明当前项目实现虽已完成正式训练和指标对照，但还未达到作者源码级或全量调参后的完整论文数值。
- `phm2012_score` 与 Jiang 论文 Score 的数值尺度仍不一致，主要用于本项目 challenge-style 惩罚解释，不作为论文 Score 数值对齐的证据。

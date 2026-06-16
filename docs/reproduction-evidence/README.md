# 论文复现真实训练证据摘要

本目录保存两篇 RUL 论文复现的可提交摘要。原始训练输出位于本机 `tmp/`，包含 `history.csv`、`metrics.json`、`predictions.csv`、完整 `comparison_metrics.csv` 和 `paper_reference_comparison.csv`，由于属于运行产物不提交仓库。本目录只保留答辩和归档需要的指标摘要。

## 证据基线

| 复现实验 | 数据来源 | 训练设置 | 原始输出目录 | 提交摘要 |
| --- | --- | --- | --- | --- |
| CNN-LSTM-AM | `real_or_provided_files` | 每轴承抽样 96 快照，relative RUL，50 epoch，batch 64 | `tmp/formal_paper_reproductions_50ep_relative/formal_cnn_lstm_attention/` | `cnn_lstm_attention_comparison_summary.csv` |
| xLSTM-Transformer | `real_or_provided_files` | 每轴承抽样 96 快照，relative RUL，附加时间索引特征，50 epoch，batch 64 | `tmp/paper_repro_xlstm_time_index_50ep/paper_xlstm_transformer/` | `xlstm_transformer_comparison_summary.csv` |

## 输出完整性

| 复现实验 | comparison rows | prediction_count | history 行数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| CNN-LSTM-AM | 4 | 92/模型 | 51/模型 | 1 行表头 + 50 行 epoch |
| xLSTM-Transformer | 18 | 87/模型 | 51/模型 | 1 行表头 + 50 行 epoch |

`huang_rul_score` 按 Huang 等论文 Eq. 11-13 计算，完美预测为 0，同一口径下越小越好。`phm2012_score` 是 challenge-style 非对称惩罚 score，不与 Huang Score 混同解释。

## 论文指标对照

- `cnn_lstm_attention_paper_reference_summary.csv` 记录 Huang 论文 Table 2/Table 5 中 RMSE 和 RMSE 改善率的对照。XJTU-SY 上 CNN-LSTM-AM normalized RMSE 为 0.146543，论文值为 0.162；PHM2012 上 CNN-LSTM-AM normalized RMSE 为 0.222228，论文值为 0.152；6/6 个论文 reference rows 在阈值内。
- `xlstm_transformer_paper_reference_summary.csv` 记录 Jiang 论文 Table 4/Table 5 中三个模型的 RMSE、R2 和 Score 对照。采用时间索引特征后，整体 reference pass rate 为 23/54；XLSTM-Transformer 在 XJTU-SY condition 1、XJTU-SY condition 3、PHM2012 condition 3 的 RMSE/R2 接近论文量级，但 PHM2012 condition 2 和 Score 尺度仍有明显差距。
- `cnn_lstm_attention_seed_sweep_summary.csv` 记录 CNN-LSTM-AM 固定种子补跑结果，用于说明 attention 结果存在随机性，不能把单次输出解释为稳定优于 baseline。
- 提交的 summary CSV 是答辩摘要，不直接作为 validator 输入。结构化完整性验证使用 `formal_reproduction_summary.json` 索引本地真实训练输出，命令如下：

```bash
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

该验证确认真实数据、50 epoch、输出文件和论文 reference pass rate 下限；它不等价于“全部指标达到作者论文结果”。

当前 selected summary 验证结果为 `paper_reference_pass_rate=0.483 (29/60)`。其中 Huang 论文部分为 6/6，Jiang xLSTM-Transformer 部分为 23/54。

如需单独复跑当前提交采用的 xLSTM time-index 输出，可使用：

```bash
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
```

## Open-Source SOTA 对照

`next goal.md` 升级后，本目录新增 open-source SOTA 目标锁定和 gap 证据：

| 文件 | 说明 |
| --- | --- |
| `open_source_sota_survey.md` | RULSurv、GNN_RUL_Benchmarking、rul-datasets/rul-adapt、RGPD reference 的调研与裁定 |
| `open_source_sota_targets.csv` | 结构化 target 表，包含 repo URL、commit、split、metric 和目标值 |
| `open_source_sota_reproduction_summary.csv` | 本地 formal 结果与 target 的 gap 表，blocked 外部 target 不填假指标 |
| `metric_driven_comparison_summary.csv` | 指标驱动汇总表，用于答辩说明 |
| `rulsurv_rsf_port/` | RULSurv RSF port 的 config、metrics、predictions 和 summary |

当前状态：

- Feature-Transformer 在 XJTU-SY condition 1 上 repeated mean normalized RMSE 为 `0.096967`，对目标 `0.0885` 的 mean gap 为 `9.57%`，达到接近强基线门槛。
- XLSTM-Transformer best observed normalized RMSE 为 `0.064558`，对目标 `0.0583` 的 best gap 为 `10.73%`，但 repeated mean gap 为 `73.40%`，仍需优化。
- RULSurv RSF port 已完成：读取 XJTU-SY `35Hz12kN` 工况 5 个轴承的 616 个原始 csv 快照，入模 611 个正 RUL 快照样本，仅排除每个轴承 `TTE=0` 的失效瞬间；在 RULSurv-compatible 25% censored row-level 5-fold CV 上，3 seeds mean true MAE 为 `10.244649` min，优于 target `12.6` min。该 row-level CV 可能把同一 bearing 的不同时间点分入不同 fold，因此不等价于 held-out-bearing 泛化；在本项目 Bearing1_3 holdout migration 上 mean true MAE 为 `19.244161` min，仍需优化。
- AutoRUL、GNN_RUL_Benchmarking 与 Weibull KIML 已锁定为 open-source SOTA/强基线 target，但当前仓库未在兼容外部环境中重跑，不能宣称这些路线已完成。

生成命令：

```bash
uv run --with scikit-survival python scripts/run_rulsurv_rsf_port.py
uv run python scripts/run_open_source_sota_evidence.py
```

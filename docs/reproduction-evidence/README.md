# 论文复现真实训练证据摘要

本目录保存两篇 RUL 论文复现的可提交摘要。原始训练输出位于本机 `tmp/`，包含 `history.csv`、`metrics.json`、`predictions.csv` 和完整 `comparison_metrics.csv`，由于属于运行产物不提交仓库。本目录只保留答辩和归档需要的指标摘要。

## 证据基线

| 复现实验 | 数据来源 | 训练设置 | 原始输出目录 | 提交摘要 |
| --- | --- | --- | --- | --- |
| CNN-LSTM-AM | `real_or_provided_files` | 每数据集抽样 48 快照，8 epoch | `tmp/paper_repro_real_metrics/paper_cnn_lstm_attention/` | `cnn_lstm_attention_comparison_summary.csv` |
| xLSTM-Transformer | `real_or_provided_files` | 每轴承抽样 16 快照，8 epoch | `tmp/paper_repro_xlstm_transformer/paper_xlstm_transformer/` | `xlstm_transformer_comparison_summary.csv` |

## 输出完整性

| 复现实验 | comparison rows | prediction_count | history 行数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| CNN-LSTM-AM | 4 | 11/模型 | 9/模型 | 1 行表头 + 8 行 epoch |
| xLSTM-Transformer | 18 | 7/模型 | 9/模型 | 1 行表头 + 8 行 epoch |

`huang_rul_score` 按 Huang 等论文 Eq. 11-13 计算，完美预测为 0，同一口径下越小越好。`phm2012_score` 是 challenge-style 非对称惩罚 score，不与 Huang Score 混同解释。


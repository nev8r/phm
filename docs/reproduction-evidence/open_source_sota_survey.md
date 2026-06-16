# Open-Source SOTA 调研与目标锁定

本文档记录 `next goal.md` 升级后的 SOTA 目标锁定结果。验收口径不再是“脚本跑通”，而是必须回答：本项目在统一指标下离可核验的开源强基线还有多远。

## 候选仓库

| 候选 | 数据集 | 可核验内容 | 代码与许可 | 本项目裁定 |
| --- | --- | --- | --- | --- |
| [AutoRUL / auto-sktime](https://github.com/Ennosigaeon/auto-sktime) | PRONOSTIA/FEMTO | AutoRUL 论文表 I 报告 PRONOSTIA RMSE `22.52 ± 5.68`，README 给出 `remaining_useful_lifetime.py femto_bearing` 复现命令 | MIT，tag `v0.1.0` commit `fe277d21104be8d2e4bd34db7ed995547007e55b` | 推荐作为 tsfresh/sklearn/sktime 路线的开源可复现强基线 target；当前未重跑 |
| [RULSurv](https://github.com/thecml/rulsurv) | XJTU-SY | README 给出数据构建、交叉验证、ISD 预测命令；论文报告 RSF 在高载荷 25% censoring 下 true MAE 为 `12.6 ± 0.8` 分钟 | MIT，commit `6365e0832de9724a5bcbbac4557c6643dfb78d91` | 已完成 Python 3.11 本地 port：RULSurv-style 特征 + scikit-survival RSF + 25% censoring + 5-fold CV，3 seeds mean true MAE `10.2738` min，超过 target；本项目 Bearing1_3 holdout migration mean `19.5107` min，仍需优化 |
| [GNN_RUL_Benchmarking](https://github.com/Frank-Wang-oss/GNN_RUL_Benchmarking) | PHM2012、XJTU-SY 等 | README 给出 `main.py` 训练命令，并提供 PHM2012/XJTU-SY 预处理入口 | GitHub 页面未显示明确 license，commit `9325667ed34976452e9323728e33a29fe0f98b5e` | 开源强基线候选；需独立环境重跑后才能作为完成证据 |
| [rul-datasets](https://github.com/tilman151/rul-datasets) | FEMTO、XJTU-SY 等 | 提供统一 RUL 数据集 LightningDataModule，支持 FEMTO/PRONOSTIA 和 XJTU-SY | pip installable，commit `f0ac3142f2fe6340e53e6158dc4f9f0ba979277a` | 数据协议候选，不是模型 SOTA |
| [rul-adapt](https://github.com/tilman151/rul-adapt) | FEMTO、XJTU-SY 等 | 提供 LSTM-DANN、ADARUL、LatentAlign、TBiGRU 等 RUL 域自适应方法 | pip installable，commit `628b6e06c99a5580f690bfad7961d4131964bbe9` | 后续迁移学习扩展候选，本轮未作为 target |
| [Weibull KIML](https://github.com/tvhahn/weibull-knowledge-informed-ml) | PRONOSTIA/FEMTO、IMS | 仓库提供 `make train_femto`、`make summarize_femto_models` 与结果汇总流程；调研记录 FEMTO `loss_rmse_test≈0.1771` | MIT，commit `c430d4b710450a1661e528675a6c1ccc64bc98e2` | 可靠性先验方向参考 target；当前未重跑 |
| [RGPD paper reference](https://arxiv.org/html/2507.09766v2) | PHM2012 | 文中表 7 报告 PHM2012 RMSE `0.0778 ± 0.0032` | 未找到可核验开源实现 | 只作参考天花板，不计入开源验收 target |

## 已锁定 Target

结构化 target 表见 `open_source_sota_targets.csv`。本轮锁定两类 target：

| 类型 | 用途 | 验收含义 |
| --- | --- | --- |
| 开源外部 target | AutoRUL、RULSurv、GNN benchmarking、Weibull KIML | 证明已找到可核验模型路线；未在当前环境重跑时不得声称完成 |
| 本地强模型 target | Jiang xLSTM-Transformer 论文目标与本项目 formal 复现 | 直接计算本项目当前结果离论文强基线的 gap |

## 当前 Gap 结论

结构化复现表见 `open_source_sota_reproduction_summary.csv`。

| target | 当前结论 |
| --- | --- |
| `jiang-xjtu-c1-feature-transformer-rmse` | 5 个 formal evidence 来源的 repeated mean `normalized_rmse=0.096967`，相对目标 `0.0885` 的 mean gap 为 `9.57%`，达到“接近”门槛 |
| `jiang-xjtu-c1-xlstm-rmse` | best observed `0.064558` 离目标 `0.0583` 的 gap 为 `10.73%`，但 repeated mean gap 为 `73.40%`，稳定性不足，状态为 `NEEDS_OPTIMIZATION` |
| `rulsurv-xjtu-high-rsf-true-mae` | RULSurv-compatible 25% censored 5-fold CV port 已完成，3 seeds mean true MAE `10.2738` min，低于 target `12.6` min，状态 `PASS`；本项目 Bearing1_3 holdout migration mean `19.5107` min，状态 `NEEDS_OPTIMIZATION` |
| `gnn-benchmark-phm2012-fc-stgnn` | 已锁定开源强基线路线，但未重跑；不能算完成 |
| `autorul-pronostia-femto-rmse` | 已锁定 MIT 许可 AutoRUL target；需独立重跑 `femto_bearing` 后才能算完成 |
| `weibull-kiml-femto-rmse` | 已锁定 MIT 许可可靠性先验 target；需独立重跑 FEMTO 流程后才能算完成 |
| `rgpd-phm2012-reference-rmse` | 参考天花板，无开源代码证据；不能作为验收 target |

## 下一步优化门槛

1. 对 xLSTM-Transformer 固定同一配置做至少 3 个 seed 的 50 epoch 重复训练，用 repeated mean 而非 best run 判定是否接近 `0.0583`。
2. 为 AutoRUL 建独立环境或容器，重跑 `femto_bearing`，把 `22.52 ± 5.68` 的 PRONOSTIA target 转成真实本地复现行。
3. 继续优化 RULSurv RSF port 的本项目 Bearing1_3 holdout split，目标从 mean true MAE `19.5107` min 降到 `15.75` min 以下，进入相对 RSF target 的 25% gap。
4. 若 xLSTM repeated mean gap 仍大于 25%，继续调 `sequence_length`、时间索引、loss、batch size、hidden size 和样本数；不得停留在“已经跑通”。

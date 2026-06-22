# Demo Script

本演示用于说明训练过程，而不是替代主线实验结论。

讲解口径：

1. 先说明 50ep 是 demo training，用于录制加速训练过程。
2. 播放视频时观察 epoch 从 1/50 到 50/50、train_loss 曲线和日志滚动。
3. 视频结尾切到对应 200ep 主线结果图。
4. 总结时只引用 200ep 作为主线结果。
5. 视频主画面不展示 val_loss / validation primary metric，避免把 demo 训练过程误读成主线性能结论。

## 视频顺序

### XJTU-SY RUL linear GRU sequence

- 视频：`video/demo_xjtu_rul_gru_50ep_accelerated.mp4`
- demo run：`demo_video_xjtu_rul_linear_gru_sequence_50ep`
- main result run：`xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep`

### XJTU-SY EarlyFault GRU sequence

- 视频：`video/demo_xjtu_early_gru_50ep_accelerated.mp4`
- demo run：`demo_video_xjtu_early_gru_sequence_50ep`
- main result run：`xjtu_main_early_gru_sequence_compact_non_label_source_200ep`

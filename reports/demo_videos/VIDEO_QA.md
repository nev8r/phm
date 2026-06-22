# 视频验收记录

统一口径：50ep 是 demo training，200ep 是 main result。

视频类型：自动生成的逐 epoch 动画，加速展示真实 50ep demo training history。

Demo 训练参数：batch_size=256，lr=0.0003，weight_decay=0.0001。

视频主画面不展示 val_loss。

视频主画面不展示 validation primary metric。

视频主画面只展示 train_loss 和滚动训练日志。

## 1. RUL 视频

- 文件名：demo_xjtu_rul_gru_50ep_accelerated.mp4
- demo run：demo_video_xjtu_rul_linear_gru_sequence_50ep
- main result run：xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep
- 任务：RUL linear regression
- 视频时长：11.30s
- 分辨率：1280x720
- 文件大小：162,842 bytes
- 50ep demo 是否完成：是（epoch=50，history=50）
- 是否加速：是，逐 epoch 动画以 10 fps 合成
- 是否展示 epoch / train_loss / 日志滚动：是
- 视频主画面不展示 val_loss：是
- 视频主画面不展示 validation primary metric：是
- 结尾是否展示 200ep training_curve：是
- 结尾是否展示 200ep true/pred by bearing：是
- 结论：通过

## 2. EarlyFault 视频

- 文件名：demo_xjtu_early_gru_50ep_accelerated.mp4
- demo run：demo_video_xjtu_early_gru_sequence_50ep
- main result run：xjtu_main_early_gru_sequence_compact_non_label_source_200ep
- 任务：EarlyFault binary classification
- 视频时长：11.30s
- 分辨率：1280x720
- 文件大小：165,781 bytes
- 50ep demo 是否完成：是（epoch=50，history=50）
- 是否加速：是，逐 epoch 动画以 10 fps 合成
- 是否展示 epoch / train_loss / 日志滚动：是
- 视频主画面不展示 val_loss：是
- 视频主画面不展示 validation primary metric：是
- 结尾是否展示 200ep training_curve：是
- 结尾是否展示 200ep confusion matrix：是
- 结论：通过

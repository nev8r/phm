# Demo Videos

本目录包含两个加速训练过程视频。

注意：

- 50ep 训练只用于视频演示训练过程。
- 200ep 结果才是主线实验结果。
- RUL 视频对应 XJTU-SY RUL linear GRU sequence。
- EarlyFault 视频对应 XJTU-SY EarlyFault GRU sequence。
- 视频是逐 epoch 动画：可以看到 epoch、训练损失和日志随时间变化。
- 为避免误读，视频主画面不展示 val_loss 或 validation primary metric；这些值仍保存在真实训练 history 中。
- Demo 训练参数：batch_size=256，lr=0.0003，weight_decay=0.0001。

## 文件

- `video/demo_xjtu_rul_gru_50ep_accelerated.mp4`：XJTU-SY RUL linear GRU sequence
- `video/demo_xjtu_early_gru_50ep_accelerated.mp4`：XJTU-SY EarlyFault GRU sequence
- `screenshots/rul_training_process.png`
- `screenshots/rul_final_figures.png`
- `screenshots/early_training_process.png`
- `screenshots/early_final_figures.png`

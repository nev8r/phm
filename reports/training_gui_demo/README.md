# 训练过程中文 GUI 演示

## 运行方式

```bash
uv run python recipes/demo/training_gui.py
```

## 数据来源

本 GUI 只读取 `reports/` 下的整理结果，不读取训练中间目录。

## 展示内容

- MLP 训练过程加速回放
- tuned MLP 结果
- XGBoost / RandomForest 预测诊断
- 特征重要性
- 最终推荐模型与 caveat

## 注意

MLP 页面的训练过程是已完成真实训练 `history.json` 的加速 replay。

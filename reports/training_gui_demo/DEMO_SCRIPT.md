# 中文训练 GUI 演示脚本

## 0:00 - 0:30 项目概览

介绍两个数据集、三个任务、模型族和 45 个真实实验。

## 0:30 - 1:40 MLP 训练回放

展示 10x 加速回放。说明读取的是已完成真实训练的 `history.json`。
视频展示的是 GUI 对已完成真实训练 history.json 的加速 replay，录屏中可以看到 epoch 进度、曲线和日志随时间变化。

## 1:40 - 2:40 调参 MLP 对比

展示 PHM2012 tuned MLP。说明 PHM2012 test 有提升，但 validation/test consistency mixed。

## 2:40 - 4:00 非 MLP 模型诊断

展示 RandomForest / XGBoost。展示 pred-vs-true、residual、confusion matrix、feature importance。

## 4:00 - 5:00 最终决策

展示推荐模型、推荐特征子集和 label-source caveat。

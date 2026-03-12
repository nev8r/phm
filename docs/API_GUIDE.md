# API 指南

## 1. 顶层 API

统一从：

```python
from USTC.SSE.BearingPrediction.api import *
```

导入常用组件。

## 2. 数据层 API

### 2.1 加载真实数据

```python
from USTC.SSE.BearingPrediction.api import XJTULoader, PHM2012Loader

xjtu_loader = XJTULoader("/path/to/XJTU-SY_Bearing_Datasets")
phm_loader = PHM2012Loader("/path/to/FEMTO")

bearing = xjtu_loader.load_entity("Bearing1_1")
```

### 2.2 加载合成数据

```python
from USTC.SSE.BearingPrediction.api import SyntheticBearingFactory

factory = SyntheticBearingFactory(random_state=42)
entity = factory.create_run_to_failure_entity("Bearing1_1")
```

## 3. 标注层 API

### 3.1 RUL 标注

```python
labeler = BearingRulLabeler(window_size=2048, stride=2048)
dataset = labeler.label(entity, "Horizontal Vibration")
train_set, test_set = dataset.split_by_ratio(0.7)
```

### 3.2 阶段标注

```python
from USTC.SSE.BearingPrediction.api import BearingStageLabeler, ThreeSigmaStageStrategy

stage_labeler = BearingStageLabeler(
    window_size=256,
    stride=256,
    stage_strategy=ThreeSigmaStageStrategy(),
)
stage_dataset, stage_result = stage_labeler.label(entity, "Horizontal Vibration")
```

### 3.3 健康指标序列标注

```python
health_dataset = HealthIndicatorLabeler(history_size=16, horizon=1).label(stage_result.health_indicator)
```

## 4. 训练 API

```python
from USTC.SSE.BearingPrediction.api import (
    BaseTrainer,
    EarlyStopping,
    TensorBoardCallback,
    GradientAlertCallback,
)

trainer = BaseTrainer(
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3),
        TensorBoardCallback("outputs/tensorboard"),
        GradientAlertCallback(),
    ],
    batch_size=16,
    max_epochs=10,
    learning_rate=1e-3,
    weight_decay=1e-4,
)
trainer.train(model, train_set, test_set)
```

## 5. 测试 API

```python
tester = BaseTester()
result = tester.test(model, test_set)
```

## 6. 预测 API

### 6.1 直接预测

```python
result = BaseTester().test(model, test_set)
```

### 6.2 不确定性预测

```python
predictor = MonteCarloDropoutPredictor(passes=10)
result = BaseTester().test(model, test_set, predictor=predictor)
```

### 6.3 滚动预测

```python
rolling_predictor = RollingPredictor()
future_values = rolling_predictor.predict_sequence(model, history, steps=5, device="cpu")
```

## 7. 评估 API

```python
evaluator = Evaluator()
evaluator.add(MAE(), MSE(), RMSE(), MAPE(), PercentError(), PHM2012Score(), NASAScore())
metrics = evaluator.evaluate(result.targets, result.predictions)
```

## 8. 可视化 API

```python
visualizer = ResultVisualizer()
visualizer.plot_prediction_curve(result.targets, result.predictions, Path("prediction_curve.png"))
visualizer.plot_confusion_matrix(targets, predictions, Path("cm.png"), labels=["healthy", "degrading", "severe"])
visualizer.plot_degradation_stages(stage_result.as_frame(), Path("stages.png"))
visualizer.plot_attention_heatmap(result.attention_weights, Path("attention.png"))
```


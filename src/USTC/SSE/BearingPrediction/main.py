"""
Project main module

this file is for orchestrating an end to end bearing prediction experiment

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

from USTC.SSE.BearingPrediction.api import (
    BaseTester,
    BaseTrainer,
    BearingRulLabeler,
    BearingStageLabeler,
    CNN,
    EarlyStopping,
    Evaluator,
    ExperimentConfig,
    ExperimentLoggerCallback,
    ExperimentTracker,
    HealthIndicatorLabeler,
    GradientAlertCallback,
    MAE,
    MAPE,
    MSE,
    MonteCarloDropoutPredictor,
    NASAScore,
    PercentError,
    PHM2008Score,
    PHM2012Score,
    RMSE,
    ResultVisualizer,
    RollingPredictor,
    SyntheticBearingFactory,
    TensorBoardCallback,
    ThreeSigmaStageStrategy,
    Transformer,
)
from USTC.SSE.BearingPrediction.common.serialization import ModelIO


def main() -> None:
    """
    run a full demonstration experiment with experiment tracking and visualization
    """

    output_root = Path("outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    synthetic_factory = SyntheticBearingFactory(random_state=42)
    train_entity = synthetic_factory.create_run_to_failure_entity("Bearing1_1", snapshot_count=48, signal_length=2048)
    test_entity = synthetic_factory.create_run_to_failure_entity("Bearing1_2", snapshot_count=36, signal_length=2048)

    rul_labeler = BearingRulLabeler(window_size=2048, stride=2048)
    train_dataset = rul_labeler.label(train_entity, "Horizontal Vibration")
    test_dataset = rul_labeler.label(test_entity, "Horizontal Vibration")
    train_set, valid_set = train_dataset.split_by_ratio(0.8)

    experiment_config = ExperimentConfig(
        run_name="cnn-rul-demo",
        dataset_name="Synthetic",
        model_name="CNN",
        optimizer_name="Adam",
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=6,
        batch_size=16,
        sampling_strategy="chronological_split",
        prediction_mode="end_to_end_direct",
        model_hyperparameters={"hidden_channels": 32, "dropout": 0.2},
        preprocessing_config={"window_size": 2048, "stride": 2048, "normalization": "zscore"},
        callback_config={"early_stopping_patience": 3, "tensorboard": True, "gradient_monitor": True},
    )
    experiment_tracker = ExperimentTracker(output_root / "experiments", experiment_config)

    model = CNN(input_size=2048, output_size=1, hidden_channels=32, dropout=0.2)
    trainer = BaseTrainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3),
            TensorBoardCallback(experiment_tracker.run_dir / "tensorboard"),
            GradientAlertCallback(vanish_threshold=1e-6, explode_threshold=50.0),
            ExperimentLoggerCallback(),
        ],
        experiment_tracker=experiment_tracker,
        batch_size=16,
        max_epochs=6,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    training_result = trainer.train(model, train_set, valid_set)

    tester = BaseTester(batch_size=32)
    direct_result = tester.test(model, test_dataset)
    uncertainty_result = tester.test(model, test_dataset, predictor=MonteCarloDropoutPredictor(passes=6))

    evaluator = Evaluator()
    evaluator.add(MAE(), MSE(), RMSE(), MAPE(), PercentError(), PHM2012Score(), PHM2008Score(), NASAScore())
    regression_metrics = evaluator.evaluate(direct_result.targets, direct_result.predictions)
    experiment_tracker.save_metrics({**regression_metrics, "best_epoch": training_result.best_epoch})
    experiment_tracker.save_predictions(uncertainty_result.as_frame())
    ModelIO.save_checkpoint(model, experiment_tracker.run_dir / "cnn_model.pt", metadata=model.get_monitor_state())

    stage_labeler = BearingStageLabeler(window_size=256, stride=256, stage_strategy=ThreeSigmaStageStrategy())
    stage_dataset, stage_result = stage_labeler.label(test_entity, "Horizontal Vibration")
    stage_train_set, stage_test_set = stage_dataset.split_by_ratio(0.8)
    stage_model = Transformer(input_size=256, output_size=3, d_model=32, nhead=4, num_layers=1, dropout=0.1, task_type="classification")
    stage_trainer = BaseTrainer(
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
        batch_size=8,
        max_epochs=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    stage_trainer.train(stage_model, stage_train_set, stage_test_set)
    stage_tester = BaseTester(batch_size=16)
    stage_test_result = stage_tester.test(stage_model, stage_test_set)

    visualizer = ResultVisualizer()
    visualizer.plot_prediction_curve(
        direct_result.targets,
        direct_result.predictions,
        experiment_tracker.run_dir / "prediction_curve.png",
        uncertainties=uncertainty_result.uncertainties,
    )
    visualizer.plot_degradation_stages(stage_result.as_frame(), experiment_tracker.run_dir / "degradation_stages.png")
    visualizer.plot_confusion_matrix(
        stage_test_result.targets,
        stage_test_result.predictions,
        experiment_tracker.run_dir / "confusion_matrix.png",
        labels=stage_result.stage_names,
    )

    attention_labeler = HealthIndicatorLabeler(history_size=16, horizon=1)
    attention_dataset = attention_labeler.label(stage_result.health_indicator)
    attention_train_set, attention_valid_set = attention_dataset.split_by_ratio(0.8)
    attention_model = Transformer(input_size=16, output_size=1, d_model=16, nhead=2, num_layers=1, dropout=0.1)
    attention_trainer = BaseTrainer(batch_size=8, max_epochs=1, learning_rate=1e-3, weight_decay=1e-4)
    attention_trainer.train(attention_model, attention_train_set, attention_valid_set)
    attention_result = tester.test(attention_model, attention_valid_set)
    if attention_result.attention_weights is not None:
        visualizer.plot_attention_heatmap(attention_result.attention_weights, experiment_tracker.run_dir / "attention_heatmap.png")

    rolling_predictor = RollingPredictor()
    rolling_forecast = rolling_predictor.predict_sequence(
        attention_model,
        attention_dataset.inputs[0],
        steps=3,
        device=trainer.device,
    )
    print({"rolling_forecast": rolling_forecast.tolist()})

    print("experiment completed")
    print(f"run_dir={experiment_tracker.run_dir}")
    print(regression_metrics)


if __name__ == "__main__":
    main()

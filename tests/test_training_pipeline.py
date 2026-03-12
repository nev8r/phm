"""
Training pipeline tests

this file is for testing trainer callbacks and experiment tracking

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
    CNN,
    EarlyStopping,
    Evaluator,
    ExperimentConfig,
    ExperimentLoggerCallback,
    ExperimentTracker,
    GradientAlertCallback,
    MAE,
    MonteCarloDropoutPredictor,
    RMSE,
    SyntheticBearingFactory,
    TensorBoardCallback,
)


def test_training_pipeline_records_experiment_artifacts(tmp_path: Path) -> None:
    factory = SyntheticBearingFactory(random_state=7)
    entity = factory.create_run_to_failure_entity("Bearing1_1", snapshot_count=20, signal_length=256)
    labeler = BearingRulLabeler(window_size=128, stride=128)
    dataset = labeler.label(entity, "Horizontal Vibration")
    train_set, valid_set = dataset.split_by_ratio(0.8)

    tracker = ExperimentTracker(
        tmp_path / "experiments",
        ExperimentConfig(
            run_name="pytest-cnn",
            dataset_name="Synthetic",
            model_name="CNN",
            optimizer_name="Adam",
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=2,
            batch_size=8,
            sampling_strategy="chronological",
            prediction_mode="direct",
        ),
    )
    model = CNN(input_size=128, output_size=1)
    trainer = BaseTrainer(
        callbacks=[
            EarlyStopping(patience=2),
            TensorBoardCallback(tracker.run_dir / "tensorboard"),
            GradientAlertCallback(warmup_steps=0, explode_threshold=0.0),
            ExperimentLoggerCallback(),
        ],
        experiment_tracker=tracker,
        batch_size=8,
        max_epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    result = trainer.train(model, train_set, valid_set)

    tester = BaseTester(batch_size=16)
    direct_result = tester.test(model, valid_set)
    uncertainty_result = tester.test(model, valid_set, predictor=MonteCarloDropoutPredictor(passes=3))

    evaluator = Evaluator().add(MAE(), RMSE())
    metrics = evaluator.evaluate(direct_result.targets, direct_result.predictions)
    tracker.save_metrics(metrics)
    tracker.save_predictions(uncertainty_result.as_frame())

    assert result.best_epoch >= 1
    assert "mae" in metrics
    assert "rmse" in metrics
    assert (tracker.run_dir / "config.yaml").exists()
    assert (tracker.run_dir / "history.csv").exists()
    assert (tracker.run_dir / "predictions.csv").exists()
    assert (tracker.run_dir / "metrics.json").exists()
    assert (tracker.run_dir / "alerts.json").exists()
    assert any(path.name.startswith("events.out.tfevents") for path in (tracker.run_dir / "tensorboard").iterdir())


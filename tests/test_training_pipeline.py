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
from USTC.SSE.BearingPrediction.common.serialization import ArtifactSerializer


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
    callbacks = [
        EarlyStopping(patience=2),
        GradientAlertCallback(warmup_steps=0, explode_threshold=0.0),
        ExperimentLoggerCallback(),
    ]
    if TensorBoardCallback.is_available():
        callbacks.insert(1, TensorBoardCallback(tracker.run_dir / "tensorboard"))

    model = CNN(input_size=128, output_size=1)
    trainer = BaseTrainer(
        callbacks=callbacks,
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
    assert "learning_rate" in result.history[0]
    assert tracker.config.callback_config["early_stopping_patience"] == 2
    assert "ExperimentLoggerCallback" in tracker.config.callback_config["callbacks"]
    if ArtifactSerializer.supports_yaml():
        assert (tracker.run_dir / "config.yaml").exists()
    assert (tracker.run_dir / "history.csv").exists()
    assert (tracker.run_dir / "predictions.csv").exists()
    assert (tracker.run_dir / "metrics.json").exists()
    assert (tracker.run_dir / "alerts.json").exists()
    if TensorBoardCallback.is_available():
        assert any(path.name.startswith("events.out.tfevents") for path in (tracker.run_dir / "tensorboard").iterdir())


def test_trainer_auto_tracker_respects_shuffle_strategy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    factory = SyntheticBearingFactory(random_state=11)
    entity = factory.create_run_to_failure_entity("Bearing1_9", snapshot_count=12, signal_length=128)
    labeler = BearingRulLabeler(window_size=64, stride=64)
    dataset = labeler.label(entity, "Horizontal Vibration")
    train_set, valid_set = dataset.split_by_ratio(0.75)

    trainer = BaseTrainer(batch_size=4, max_epochs=1, learning_rate=1e-3, weight_decay=1e-4, shuffle_train=False)
    trainer.train(CNN(input_size=64, output_size=1), train_set, valid_set)

    assert trainer.experiment_tracker is not None
    assert trainer.experiment_tracker.config.sampling_strategy == "chronological"


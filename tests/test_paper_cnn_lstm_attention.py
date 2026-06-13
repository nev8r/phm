"""
CNN-LSTM attention paper reproduction tests

this file is for testing feature sequence labeling and CNN-LSTM-AM model

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from USTC.SSE.BearingPrediction.api import (
    BaseTester,
    BaseTrainer,
    CNNLSTMAttention,
    FeatureSequenceRulLabeler,
    SyntheticBearingFactory,
)
from USTC.SSE.BearingPrediction.examples.demo_workflows import (
    create_demo_phm2012_dataset,
    create_demo_xjtu_dataset,
    run_paper_cnn_lstm_attention_reproduction,
)
from USTC.SSE.BearingPrediction.feature import FeatureConfig, SignalFeatureExtractor


def test_default_signal_features_match_paper_feature_count_and_names() -> None:
    signal_values = torch.linspace(-1.0, 1.0, 128).numpy()
    extractor = SignalFeatureExtractor(FeatureConfig(sample_rate=25_600.0))

    features = extractor.extract_one(signal_values)

    assert len(FeatureConfig(sample_rate=25_600.0).enabled_features) == 19
    assert len(features) == 19
    assert {
        "absolute_mean",
        "peak_to_peak",
        "shape_factor",
        "impulse_factor",
        "margin_factor",
        "clearance_factor",
        "spectral_centroid",
        "spectral_rms_frequency",
    }.issubset(features)


def test_feature_sequence_labeler_builds_normalized_paper_feature_sequences() -> None:
    factory = SyntheticBearingFactory(random_state=19)
    entity = factory.create_run_to_failure_entity("Bearing1_1", snapshot_count=8, signal_length=128)
    labeler = FeatureSequenceRulLabeler(sequence_length=3, window_size=64, stride=64)

    dataset = labeler.label(entity, "Horizontal Vibration")

    assert dataset.inputs.shape == (6, 3, 19)
    assert dataset.targets.tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    assert dataset.metadata_frame.iloc[0]["start_sample_index"] == 0
    assert dataset.metadata_frame.iloc[0]["end_sample_index"] == 2
    assert dataset.input_name == "feature_sequence"


def test_cnn_lstm_attention_default_architecture_matches_paper_depths() -> None:
    model = CNNLSTMAttention(feature_size=19, output_size=1)

    convolution_count = sum(1 for layer in model.feature_encoder if isinstance(layer, torch.nn.Conv1d))
    pooling_count = sum(1 for layer in model.feature_encoder if isinstance(layer, torch.nn.MaxPool1d))
    head_linear_count = sum(1 for layer in model.head if isinstance(layer, torch.nn.Linear))

    assert convolution_count == 3
    assert pooling_count == 3
    assert model.temporal_encoder.num_layers == 3
    assert head_linear_count == 3
    assert model.use_attention is True


def test_cnn_lstm_attention_forward_returns_prediction_and_attention() -> None:
    model = CNNLSTMAttention(feature_size=19, output_size=1, cnn_channels=8, lstm_hidden_size=12, lstm_layers=1)
    inputs = torch.randn(4, 5, 19)

    output = model(inputs)
    attention_weights = model.maybe_get_attention()

    assert output["prediction"].shape == (4, 1)
    assert attention_weights is not None
    assert attention_weights.shape == (4, 5)
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_cnn_lstm_attention_trains_with_existing_trainer() -> None:
    factory = SyntheticBearingFactory(random_state=23)
    entity = factory.create_run_to_failure_entity("Bearing1_2", snapshot_count=10, signal_length=128)
    dataset = FeatureSequenceRulLabeler(sequence_length=3, window_size=64, stride=64).label(
        entity,
        "Horizontal Vibration",
    )
    train_set, valid_set = dataset.split_by_ratio(0.75)
    model = CNNLSTMAttention(feature_size=dataset.inputs.shape[-1], output_size=1, cnn_channels=4, lstm_hidden_size=8, lstm_layers=1)
    trainer = BaseTrainer(device="cpu", batch_size=4, max_epochs=1, learning_rate=1e-3, weight_decay=1e-4, shuffle_train=False)

    result = trainer.train(model, train_set, valid_set)

    assert result.best_epoch == 1
    assert "val_rmse" in result.history[-1]


def test_base_tester_collects_attention_for_every_prediction() -> None:
    factory = SyntheticBearingFactory(random_state=29)
    entity = factory.create_run_to_failure_entity("Bearing2_1", snapshot_count=12, signal_length=128)
    dataset = FeatureSequenceRulLabeler(sequence_length=3, window_size=64, stride=64).label(
        entity,
        "Horizontal Vibration",
    )
    train_set, valid_set = dataset.split_by_ratio(0.5)
    model = CNNLSTMAttention(feature_size=dataset.inputs.shape[-1], output_size=1, cnn_channels=4, lstm_hidden_size=8, lstm_layers=1)
    trainer = BaseTrainer(device="cpu", batch_size=3, max_epochs=1, learning_rate=1e-3, weight_decay=1e-4, shuffle_train=False)
    trainer.train(model, train_set, valid_set)

    result = BaseTester(device="cpu", batch_size=2).test(model, valid_set)

    assert result.attention_weights is not None
    assert result.attention_weights.shape == (len(valid_set), 3)


def test_paper_reproduction_workflow_trains_two_datasets_and_attention_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BEARING_EXAMPLE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("BEARING_EXAMPLE_EPOCHS", "1")
    xjtu_root = create_demo_xjtu_dataset(tmp_path / "input_data", sample_count=12, signal_length=128)
    phm_root = create_demo_phm2012_dataset(tmp_path / "input_data", sample_count=12, signal_length=128)

    result = run_paper_cnn_lstm_attention_reproduction(
        xjtu_root=xjtu_root,
        phm2012_root=phm_root,
        max_samples_per_entity=12,
        prefer_real_data=True,
    )

    comparison_frame = pd.read_csv(result["comparison_path"])
    required_metric_columns = {
        "huang_rul_score",
        "normalized_rmse",
        "smape",
        "over_prediction_rate",
        "within_10_percent_rate",
        "rmse_reduction_pct",
        "huang_score_change_pct",
    }
    assert set(comparison_frame["dataset_name"]) == {"XJTU-SY", "PHM2012"}
    assert set(comparison_frame["model_name"]) == {"CNN-LSTM-AM", "CNN-LSTM"}
    assert required_metric_columns.issubset(comparison_frame.columns)
    assert comparison_frame["huang_rul_score"].notna().all()
    assert comparison_frame["normalized_rmse"].notna().all()
    assert (comparison_frame["epoch_count"] == 1).all()
    assert Path(result["comparison_path"]).exists()
    assert result["used_dataset_count"] == 2
    assert result["trained_model_count"] == 4

    for run_summary in result["runs"]:
        prediction_frame = pd.read_csv(run_summary["prediction_path"])
        attention_frame = pd.read_csv(run_summary["attention_path"])
        metrics = json.loads(Path(run_summary["metrics_path"]).read_text(encoding="utf-8"))
        assert len(prediction_frame) == run_summary["prediction_count"]
        assert "huang_rul_score" in metrics
        assert "normalized_rmse" in metrics
        if run_summary["model_name"] == "CNN-LSTM-AM":
            assert len(attention_frame) == run_summary["prediction_count"]
        assert Path(run_summary["history_path"]).exists()

"""
Feature backend tests

this file is for testing pluggable feature extraction backends

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from USTC.SSE.BearingPrediction.api import BearingRulLabeler, FeatureSequenceRulLabeler, SyntheticBearingFactory
from USTC.SSE.BearingPrediction.feature import FeatureBackendConfig, FeatureConfig, create_feature_backend


def _demo_windows() -> list[np.ndarray]:
    base_signal = np.linspace(-1.0, 1.0, 64, dtype=float)
    return [
        base_signal,
        np.sin(np.linspace(0.0, 2.0 * np.pi, 64, dtype=float)),
        np.cos(np.linspace(0.0, 2.0 * np.pi, 64, dtype=float)),
    ]


def test_manual_feature_backend_matches_default_feature_config() -> None:
    backend = create_feature_backend("manual_19", sample_rate=25_600.0)

    feature_frame = backend.extract(_demo_windows())

    expected_columns = list(FeatureConfig(sample_rate=25_600.0).enabled_features)
    assert feature_frame.shape == (3, 19)
    assert feature_frame.columns.tolist() == expected_columns
    assert np.isfinite(feature_frame.to_numpy(dtype=float)).all()


def test_feature_backend_config_accepts_name_alias() -> None:
    config = FeatureBackendConfig(name="manual_19", sample_rate=12_000.0)
    backend = create_feature_backend(config)

    feature_frame = backend.extract(_demo_windows()[:1])

    assert config.name == "manual_19"
    assert feature_frame.shape == (1, 19)


def test_composite_feature_backend_prefixes_tsfresh_columns() -> None:
    if find_spec("tsfresh") is None:
        pytest.skip("tsfresh advanced extra is not installed")

    backend = create_feature_backend("manual_19_plus_tsfresh_minimal", sample_rate=25_600.0)

    feature_frame = backend.extract(_demo_windows())

    manual_columns = list(FeatureConfig(sample_rate=25_600.0).enabled_features)
    assert feature_frame.shape[0] == 3
    assert manual_columns == feature_frame.columns[:19].tolist()
    assert any(column_name.startswith("tsfresh__") for column_name in feature_frame.columns[19:])
    assert np.isfinite(feature_frame.to_numpy(dtype=float)).all()


def test_tsfresh_feature_backend_reports_advanced_extra_when_missing() -> None:
    if find_spec("tsfresh") is not None:
        pytest.skip("tsfresh advanced extra is installed")

    backend = create_feature_backend("tsfresh_minimal", sample_rate=25_600.0)

    with pytest.raises(RuntimeError, match="uv run --extra advanced"):
        backend.extract(_demo_windows())


def test_feature_sequence_labeler_manual_backend_keeps_default_shape() -> None:
    factory = SyntheticBearingFactory(random_state=31)
    entity = factory.create_run_to_failure_entity("Bearing1_1", snapshot_count=8, signal_length=128)
    labeler = FeatureSequenceRulLabeler(sequence_length=3, window_size=64, stride=64, feature_backend="manual_19")

    dataset = labeler.label(entity, "Horizontal Vibration")

    assert dataset.inputs.shape == (6, 3, 19)
    assert dataset.feature_frame is not None
    assert dataset.feature_frame.shape == (6, 19)


def test_bearing_rul_labeler_feature_inputs_match_selected_backend_columns() -> None:
    factory = SyntheticBearingFactory(random_state=37)
    entity = factory.create_run_to_failure_entity("Bearing1_2", snapshot_count=4, signal_length=128)
    labeler = BearingRulLabeler(
        window_size=64,
        stride=64,
        input_representation="features",
        feature_backend="manual_19",
    )

    dataset = labeler.label(entity, "Horizontal Vibration")

    assert dataset.feature_frame is not None
    assert dataset.inputs.shape[1] == dataset.feature_frame.shape[1]
    assert dataset.feature_frame.shape[1] == 19

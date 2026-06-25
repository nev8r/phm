"""
Test Stage 4 task manifest window construction.

Purpose: verify test stage 4 task manifest window construction behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.task.WindowBuilder import WindowBuilder


def _features(count=5, bearing_id="Bearing1_1"):
    return pd.DataFrame({
        "sample_uid": [f"{bearing_id}_{i}" for i in range(count)],
        "dataset": ["XJTU-SY"] * count,
        "bearing_id": [bearing_id] * count,
        "condition_id": ["35Hz12kN"] * count,
        "source_group": [None] * count,
        "sample_id": [f"{i:06d}" for i in range(count)],
        "timestep": list(range(count)),
        "f1": [float(i) for i in range(count)],
        "f2": [float(i + 10) for i in range(count)],
    })


def _labels(features):
    return features[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep"]].assign(
        piecewise_rul_norm=[1.0 - (i / max(len(features) - 1, 1)) for i in range(len(features))]
    )


def test_window_builder_creates_one_tabular_example_per_sample():
    features = _features(count=5)
    labels = _labels(features)
    cfg = OmegaConf.create({"input_mode": "tabular"})

    manifest = WindowBuilder().build(features, labels, split_result=None, cfg=cfg)

    assert len(manifest) == 5
    assert set(manifest["split"]) == {"all"}
    assert manifest["num_timesteps"].tolist() == [1, 1, 1, 1, 1]
    assert (manifest["target_sample_uid"] == manifest["start_sample_uid"]).all()
    assert (manifest["target_sample_uid"] == manifest["end_sample_uid"]).all()


def test_window_builder_creates_feature_sequence_windows_within_bearing():
    features = _features(count=5)
    labels = _labels(features)
    cfg = OmegaConf.create({
        "input_mode": "feature_sequence",
        "sequence": {
            "length": 3,
            "stride": 1,
            "target_position": "last",
            "drop_incomplete": True,
            "allow_cross_bearing": False,
            "allow_cross_split": False,
        },
    })

    manifest = WindowBuilder().build(features, labels, split_result=None, cfg=cfg)

    assert len(manifest) == 3
    assert manifest["start_timestep"].tolist() == [0, 1, 2]
    assert manifest["end_timestep"].tolist() == [2, 3, 4]
    assert manifest["target_sample_uid"].tolist() == ["Bearing1_1_2", "Bearing1_1_3", "Bearing1_1_4"]
    assert manifest["num_timesteps"].tolist() == [3, 3, 3]


def test_window_builder_never_crosses_bearing_for_sequence_windows():
    features = pd.concat([_features(count=3, bearing_id="Bearing1_1"), _features(count=3, bearing_id="Bearing1_2")])
    labels = _labels(features.reset_index(drop=True))
    cfg = OmegaConf.create({
        "input_mode": "feature_sequence",
        "sequence": {"length": 3, "stride": 1, "drop_incomplete": True},
    })

    manifest = WindowBuilder().build(features, labels, split_result=None, cfg=cfg)

    assert len(manifest) == 2
    assert manifest["bearing_id"].tolist() == ["Bearing1_1", "Bearing1_2"]
    assert all(len(set(row.split("|"))) == 3 for row in manifest["window_sample_uids"])

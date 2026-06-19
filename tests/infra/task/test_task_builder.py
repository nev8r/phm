"""
Test Stage 4 task builder.
"""

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult
from USTC.SSE.BearingPrediction.infra.task.TaskBuilder import TaskBuilder


def _features():
    rows = []
    for bearing_id in ["Bearing1_1", "Bearing1_2", "Bearing1_3"]:
        for timestep in range(4):
            rows.append({
                "sample_uid": f"{bearing_id}_{timestep}",
                "dataset": "XJTU-SY",
                "bearing_id": bearing_id,
                "condition_id": "35Hz12kN",
                "source_group": None,
                "sample_id": f"{timestep:06d}",
                "timestep": timestep,
                "f1": float(timestep),
                "f2": float(timestep + 100),
                "drop_me": float(timestep + 200),
            })
    return pd.DataFrame(rows)


def _labels(features):
    labels = features[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep"]].copy()
    labels["piecewise_rul_norm"] = labels.groupby("bearing_id").cumcount().map({0: 1.0, 1: 0.66, 2: 0.33, 3: 0.0})
    labels["health_state_id"] = labels["timestep"].clip(upper=3)
    labels["health_state_name"] = labels["health_state_id"].map({0: "healthy", 1: "slight", 2: "moderate", 3: "severe"})
    return labels


def _split():
    return SplitResult(
        name="toy",
        train_sample_uids=[f"Bearing1_1_{i}" for i in range(4)],
        val_sample_uids=[f"Bearing1_2_{i}" for i in range(4)],
        test_sample_uids=[f"Bearing1_3_{i}" for i in range(4)],
        train_bearings=["Bearing1_1"],
        val_bearings=["Bearing1_2"],
        test_bearings=["Bearing1_3"],
    )


def test_task_builder_creates_tabular_regression_datamodule_with_split():
    features = _features()
    labels = _labels(features)
    cfg = OmegaConf.create({
        "name": "rul_tabular",
        "version": "v1",
        "task_type": "regression",
        "input_mode": "tabular",
        "feature_source": "cleaned",
        "feature_columns": {"include": "all", "exclude_columns": ["drop_me"]},
        "target": {"columns": ["piecewise_rul_norm"], "dtype": "float32"},
    })

    datamodule = TaskBuilder(cfg).build(features, labels, split_result=_split())

    assert datamodule.input_dim == 2
    assert datamodule.output_dim == 1
    assert len(datamodule.task_manifest) == len(features)
    assert len(datamodule.train) == 4
    assert len(datamodule.val) == 4
    assert len(datamodule.test) == 4
    assert datamodule.task_report["num_train_examples"] == 4
    assert datamodule.task_spec["feature_columns"] == ["f1", "f2"]


def test_task_builder_creates_sequence_regression_dataset():
    features = _features()
    labels = _labels(features)
    cfg = OmegaConf.create({
        "name": "rul_sequence",
        "version": "v1",
        "task_type": "regression",
        "input_mode": "feature_sequence",
        "feature_source": "cleaned",
        "feature_columns": {"include": "all", "exclude_columns": ["drop_me"]},
        "sequence": {"length": 3, "stride": 1, "target_position": "last", "drop_incomplete": True},
        "target": {"columns": ["piecewise_rul_norm"], "dtype": "float32"},
    })

    datamodule = TaskBuilder(cfg).build(features, labels, split_result=_split())
    item = datamodule.train[0]

    assert len(datamodule.task_manifest) == 6
    assert item["x"].shape == (3, 2)
    assert item["y"].shape == (1,)
    assert item["target_sample_uid"] == "Bearing1_1_2"
    assert datamodule.task_report["sequence"]["length"] == 3


def test_task_builder_reports_class_distribution_for_classification():
    features = _features()
    labels = _labels(features)
    cfg = OmegaConf.create({
        "name": "health_state_tabular",
        "version": "v1",
        "task_type": "multiclass_classification",
        "input_mode": "tabular",
        "feature_source": "cleaned",
        "feature_columns": {"include": "all", "exclude_columns": ["drop_me"]},
        "target": {"columns": ["health_state_id"], "dtype": "int64", "num_classes": 4},
    })

    datamodule = TaskBuilder(cfg).build(features, labels, split_result=_split())

    assert str(datamodule.train[0]["y"].dtype) == "torch.int64"
    assert datamodule.task_report["class_distribution"]["train"]
    assert datamodule.task_spec["target_columns"] == ["health_state_id"]

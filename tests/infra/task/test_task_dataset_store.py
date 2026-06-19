"""
Test Stage 4 dataset, data module, and store behavior.
"""

import json

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.task.DataModule import DataModule
from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset
from USTC.SSE.BearingPrediction.infra.task.TaskStore import TaskStore


def _frames():
    features = pd.DataFrame({
        "sample_uid": ["s0", "s1", "s2"],
        "dataset": ["XJTU-SY"] * 3,
        "bearing_id": ["Bearing1_1"] * 3,
        "condition_id": ["35Hz12kN"] * 3,
        "source_group": [None] * 3,
        "sample_id": ["000000", "000001", "000002"],
        "timestep": [0, 1, 2],
        "f1": [0.1, 0.2, 0.3],
        "f2": [1.1, 1.2, 1.3],
    })
    labels = features[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep"]].copy()
    labels["piecewise_rul_norm"] = [1.0, 0.5, 0.0]
    labels["early_fault"] = [0, 1, 1]
    manifest = pd.DataFrame({
        "example_uid": ["all::Bearing1_1::000000-000000", "all::Bearing1_1::000000-000002"],
        "split": ["all", "all"],
        "dataset": ["XJTU-SY", "XJTU-SY"],
        "bearing_id": ["Bearing1_1", "Bearing1_1"],
        "condition_id": ["35Hz12kN", "35Hz12kN"],
        "source_group": [None, None],
        "start_sample_uid": ["s0", "s0"],
        "end_sample_uid": ["s0", "s2"],
        "target_sample_uid": ["s0", "s2"],
        "start_timestep": [0, 0],
        "end_timestep": [0, 2],
        "target_timestep": [0, 2],
        "num_timesteps": [1, 3],
        "window_sample_uids": ["s0", "s0|s1|s2"],
    })
    return features, labels, manifest


def test_task_dataset_returns_tabular_and_sequence_tensors_with_metadata():
    features, labels, manifest = _frames()
    tabular = TaskDataset(features, labels, manifest.iloc[[0]], ["f1", "f2"], ["piecewise_rul_norm"], "tabular", "regression")
    sequence = TaskDataset(features, labels, manifest.iloc[[1]], ["f1", "f2"], ["early_fault"], "feature_sequence", "binary_classification")

    tabular_item = tabular[0]
    sequence_item = sequence[0]

    assert tabular_item["x"].shape == (2,)
    assert tabular_item["y"].shape == (1,)
    assert sequence_item["x"].shape == (3, 2)
    assert str(sequence_item["y"].dtype) == "torch.int64"
    assert sequence_item["example_uid"] == "all::Bearing1_1::000000-000002"
    assert sequence_item["sample_uid"] == "s2"
    assert tabular_item["split"] == "all"
    assert tabular_item["dataset"] == "XJTU-SY"
    assert tabular_item["condition_id"] == "35Hz12kN"
    assert tabular_item["target_timestep"] == 0
    assert sequence_item["split"] == "all"
    assert sequence_item["dataset"] == "XJTU-SY"
    assert sequence_item["condition_id"] == "35Hz12kN"
    assert sequence_item["target_timestep"] == 2


def test_data_module_exposes_split_mapping_and_dimensions():
    features, labels, manifest = _frames()
    dataset = TaskDataset(features, labels, manifest.iloc[[0]], ["f1", "f2"], ["piecewise_rul_norm"], "tabular", "regression")
    datamodule = DataModule(
        train=dataset,
        val=None,
        test=None,
        all=None,
        task_manifest=manifest,
        feature_columns=["f1", "f2"],
        target_columns=["piecewise_rul_norm"],
        task_spec={"name": "toy"},
        task_report={"ok": True},
    )

    assert datamodule.input_dim == 2
    assert datamodule.output_dim == 1
    assert datamodule.splits() == {"train": dataset}


def test_task_store_writes_task_artifacts(tmp_path):
    _, _, manifest = _frames()
    spec = {"name": "toy", "hash": "abc"}
    report = {"ok": True}

    TaskStore(ArtifactManager(tmp_path), write_csv=True).save(
        manifest=manifest,
        task_spec=spec,
        task_report=report,
        feature_columns=["f1", "f2"],
        target_columns=["piecewise_rul_norm"],
    )

    assert (tmp_path / "task" / "task_manifest.parquet").exists()
    assert (tmp_path / "task" / "task_manifest.csv").exists()
    assert json.loads((tmp_path / "task" / "task_spec.json").read_text())["name"] == "toy"
    assert (tmp_path / "task" / "feature_columns.txt").read_text().splitlines() == ["f1", "f2"]
    assert (tmp_path / "task" / "target_columns.txt").read_text().splitlines() == ["piecewise_rul_norm"]

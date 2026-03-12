"""
Data io and loader tests

this file is for testing dataset serialization and loader compatibility

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from USTC.SSE.BearingPrediction.api import BearingRulLabeler, PHM2012Loader, SyntheticBearingFactory, XJTULoader
from USTC.SSE.BearingPrediction.common import serialization as serialization_module
from USTC.SSE.BearingPrediction.common.serialization import ArtifactSerializer
from USTC.SSE.BearingPrediction.training import ExperimentConfig, ExperimentTracker


def test_dataset_export_and_loader_parse(tmp_path: Path) -> None:
    factory = SyntheticBearingFactory(random_state=3)
    entity = factory.create_run_to_failure_entity("Bearing1_3", snapshot_count=12, signal_length=128)
    labeler = BearingRulLabeler(window_size=64, stride=64)
    dataset = labeler.label(entity, "Horizontal Vibration")

    dataset.export(tmp_path / "dataset.csv")
    dataset.export(tmp_path / "dataset.pkl")
    csv_frame = ArtifactSerializer.load_dataframe(tmp_path / "dataset.csv")
    pkl_frame = ArtifactSerializer.load_dataframe(tmp_path / "dataset.pkl")

    assert not csv_frame.empty
    assert not pkl_frame.empty

    dataset_root = tmp_path / "XJTU-SY_Bearing_Datasets" / "35Hz12kN" / "Bearing1_1"
    dataset_root.mkdir(parents=True)
    sample_frame = pd.DataFrame({0: np.arange(64), 1: np.arange(64) * 2})
    sample_frame.to_csv(dataset_root / "1.csv", index=False, header=False)
    sample_frame.to_csv(dataset_root / "2.csv", index=False, header=False)

    loader = XJTULoader(tmp_path / "XJTU-SY_Bearing_Datasets")
    loaded_entity = loader.load_entity("Bearing1_1")

    assert loaded_entity.entity_id == "Bearing1_1"
    assert "Horizontal Vibration" in loaded_entity.channel_names()


def test_xjtu_loader_skips_header_rows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "XJTU-SY_Bearing_Datasets" / "35Hz12kN" / "Bearing1_1"
    dataset_root.mkdir(parents=True)
    sample_frame = pd.DataFrame(
        {
            "Horizontal_vibration_signals": np.arange(8, dtype=float),
            "Vertical_vibration_signals": np.arange(8, dtype=float) * 2.0,
        }
    )
    sample_frame.to_csv(dataset_root / "1.csv", index=False)
    sample_frame.to_csv(dataset_root / "2.csv", index=False)

    loader = XJTULoader(tmp_path / "XJTU-SY_Bearing_Datasets")
    loaded_entity = loader.load_entity("Bearing1_1")

    assert loaded_entity.samples.shape[0] == 2
    assert loaded_entity.samples.iloc[0]["source_file"] == "1.csv"
    assert np.allclose(loaded_entity.samples.iloc[0]["Horizontal Vibration"][:3], [0.0, 1.0, 2.0])


def test_phm2012_loader_preserves_temperature_channel(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Training_set" / "Learning_set" / "Bearing1_1"
    dataset_root.mkdir(parents=True)
    acceleration_frame = pd.DataFrame(
        {
            0: [9, 9, 9],
            1: [39, 39, 39],
            2: [39, 39, 39],
            3: [65664, 65703, 65742],
            4: [0.552, 0.501, 0.138],
            5: [-0.146, -0.48, 0.435],
        }
    )
    temperature_frame = pd.DataFrame({0: [9, 9], 1: [40, 40], 2: [47, 47], 3: [5, 6], 4: [70.378, 70.397]})
    acceleration_frame.to_csv(dataset_root / "acc_00001.csv", index=False, header=False)
    acceleration_frame.to_csv(dataset_root / "acc_00002.csv", index=False, header=False)
    temperature_frame.to_csv(dataset_root / "temp_00001.csv", index=False, header=False)

    loader = PHM2012Loader(tmp_path)
    loaded_entity = loader.load_entity("Bearing1_1")

    assert loaded_entity.samples.shape[0] == 2
    assert set(loaded_entity.samples["source_file"]) == {"acc_00001.csv", "acc_00002.csv"}
    assert "Temperature" in loaded_entity.channel_names()
    assert loaded_entity.samples.iloc[0]["temperature_file"] == "temp_00001.csv"
    assert np.allclose(loaded_entity.samples.iloc[0]["Temperature"], [70.378, 70.397])


def test_yaml_outputs_become_optional_when_pyyaml_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(serialization_module, "yaml", None)

    tracker = ExperimentTracker(
        tmp_path / "experiments",
        ExperimentConfig(
            run_name="yaml-optional",
            dataset_name="Synthetic",
            model_name="CNN",
            optimizer_name="Adam",
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=1,
            batch_size=4,
            sampling_strategy="chronological",
            prediction_mode="direct",
        ),
    )
    tracker.save_metrics({"rmse": 0.1})

    assert ArtifactSerializer.supports_yaml() is False
    assert (tracker.run_dir / "config.json").exists()
    assert not (tracker.run_dir / "config.yaml").exists()
    assert (tracker.run_dir / "metrics.json").exists()
    assert not (tracker.run_dir / "metrics.yaml").exists()

    with pytest.raises(RuntimeError, match="PyYAML"):
        ArtifactSerializer.save_object({"rmse": 0.1}, tmp_path / "metrics.yaml")

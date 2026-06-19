"""
Test the Stage 3 build_labels CLI.
"""

import json
import shutil
import subprocess

import pandas as pd

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root


def test_cli_build_labels_linear_rul_without_features(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=build_labels",
            "dataset=xjtu_sy",
            "split=none",
            "feature=none",
            "label=linear_rul",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = sorted((artifact_root / "runs").iterdir())[0]
    labels = pd.read_parquet(run_dir / "labels" / "labels.parquet")
    report = json.loads((run_dir / "labels" / "label_report.json").read_text())

    assert "linear_rul_norm" in labels.columns
    assert labels["sample_uid"].is_unique
    assert labels["linear_rul_norm"].between(0, 1).all()
    assert not (run_dir / "features").exists()
    assert not (run_dir / "hi").exists()
    assert report["hi_enabled"] is False


def test_cli_build_labels_degradation_basic_writes_hi_and_labels(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=build_labels",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
            "split.condition_id=35Hz12kN",
            "split.test_bearing_id=Bearing1_5",
            "split.val_bearing_id=Bearing1_4",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = sorted((artifact_root / "runs").iterdir())[0]
    labels = pd.read_parquet(run_dir / "labels" / "labels.parquet")
    fpt = json.loads((run_dir / "hi" / "fpt.json").read_text())

    assert (run_dir / "features" / "cleaned_features.parquet").exists()
    assert (run_dir / "hi" / "hi.parquet").exists()
    assert (run_dir / "labels" / "label_spec.json").exists()
    for column in ["linear_rul_norm", "piecewise_rul_norm", "health_state_id", "health_state_name", "early_fault"]:
        assert column in labels.columns
    assert labels["piecewise_rul_norm"].between(0, 1).all()
    assert set(labels["early_fault"]).issubset({0, 1})
    assert len(fpt["results"]) == labels["bearing_id"].nunique()


def test_cli_build_labels_degradation_basic_for_phm2012(tmp_path):
    dataset_root = create_fake_phm2012_root(tmp_path / "phm2012")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=build_labels",
            "dataset=phm2012",
            "split=none",
            "feature=manual_basic",
            "label=degradation_basic",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = sorted((artifact_root / "runs").iterdir())[0]
    labels = pd.read_parquet(run_dir / "labels" / "labels.parquet")

    assert "fault_type_stage_name" not in labels.columns
    assert "health_state_name" in labels.columns
    assert (run_dir / "hi" / "fpt.json").exists()

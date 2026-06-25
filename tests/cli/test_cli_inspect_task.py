"""
Test the Stage 4 inspect_task CLI.

Purpose: verify test the stage 4 inspect_task cli behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import json
import shutil
import subprocess

import pandas as pd

from tests.infra.dataset_fixtures import create_fake_xjtu_root


def _run_dir(artifact_root):
    return sorted((artifact_root / "runs").iterdir())[0]


def test_cli_inspect_task_rul_tabular(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=inspect_task",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=rul_tabular",
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
    run_dir = _run_dir(artifact_root)
    report = json.loads((run_dir / "task" / "task_report.json").read_text())

    assert (run_dir / "task" / "task_manifest.parquet").exists()
    assert (run_dir / "task" / "task_spec.json").exists()
    assert report["task_type"] == "regression"
    assert report["input_mode"] == "tabular"
    assert report["num_train_examples"] > 0


def test_cli_inspect_task_rul_sequence(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=inspect_task",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=rul_sequence",
            "task.sequence.length=2",
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
    run_dir = _run_dir(artifact_root)
    report = json.loads((run_dir / "task" / "task_report.json").read_text())
    manifest = pd.read_parquet(run_dir / "task" / "task_manifest.parquet")

    assert report["input_mode"] == "feature_sequence"
    assert report["sequence"]["length"] == 2
    assert manifest["num_timesteps"].eq(2).all()
    assert (manifest["start_sample_uid"].str.split("::").str[1] == manifest["end_sample_uid"].str.split("::").str[1]).all()


def test_cli_inspect_task_health_state_tabular(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=inspect_task",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=health_state_tabular",
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
    run_dir = _run_dir(artifact_root)
    report = json.loads((run_dir / "task" / "task_report.json").read_text())
    target_columns = (run_dir / "task" / "target_columns.txt").read_text().splitlines()

    assert report["task_type"] == "multiclass_classification"
    assert report["class_distribution"]
    assert target_columns == ["health_state_id"]

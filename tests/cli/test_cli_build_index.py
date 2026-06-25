"""
Test the Stage 1 build_index CLI.

Purpose: verify test the stage 1 build_index cli behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import json
import shutil
import subprocess

import pandas as pd

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root


def test_cli_build_index_writes_xjtu_index_and_split_artifacts(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=build_index",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
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
    index = pd.read_parquet(run_dir / "index" / "sample_index.parquet")
    index_report = json.loads((run_dir / "index" / "index_report.json").read_text())
    split = json.loads((run_dir / "split" / "split.json").read_text())
    split_report = json.loads((run_dir / "split" / "split_report.json").read_text())

    assert len(index) == 7
    assert (run_dir / "index" / "sample_index.csv").exists()
    assert index_report["ok"] is True
    assert split["test_bearings"] == ["Bearing1_5"]
    assert split_report["ok"] is True


def test_cli_build_index_writes_phm2012_index_without_split(tmp_path):
    dataset_root = create_fake_phm2012_root(tmp_path / "phm2012")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=build_index",
            "dataset=phm2012",
            "split=none",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = sorted((artifact_root / "runs").iterdir())[0]
    index = pd.read_csv(run_dir / "index" / "sample_index.csv")
    index_report = json.loads((run_dir / "index" / "index_report.json").read_text())

    assert len(index) == 6
    assert "Bearing1_4" in set(index["bearing_id"])
    assert index_report["dataset"] == "PHM2012"
    assert not (run_dir / "split").exists()

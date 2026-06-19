"""
Test the Stage 2 extract_features CLI.
"""

import json
import shutil
import subprocess

import pandas as pd

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root


def test_cli_extract_features_writes_manual_feature_artifacts_with_split(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=extract_features",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
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
    raw = pd.read_parquet(run_dir / "features" / "raw_features.parquet")
    cleaned = pd.read_parquet(run_dir / "features" / "cleaned_features.parquet")
    report = json.loads((run_dir / "features" / "feature_report.json").read_text())

    assert (run_dir / "index" / "sample_index.parquet").exists()
    assert (run_dir / "split" / "split.json").exists()
    assert (run_dir / "features" / "feature_spec.json").exists()
    assert (run_dir / "features" / "cleaner_state.pkl").exists()
    assert len(raw) == 7
    assert len(cleaned) == 7
    assert report["ok"] is True
    assert report["cleaner_fit_scope"] == "train_only"


def test_cli_extract_features_writes_tsfresh_features_without_split(tmp_path):
    dataset_root = create_fake_phm2012_root(tmp_path / "phm2012")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=extract_features",
            "dataset=phm2012",
            "split=none",
            "feature=tsfresh_minimal",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = sorted((artifact_root / "runs").iterdir())[0]
    report = json.loads((run_dir / "features" / "feature_report.json").read_text())
    cleaned = pd.read_parquet(run_dir / "features" / "cleaned_features.parquet")

    assert not (run_dir / "split").exists()
    assert any(column.startswith("tsfresh__h__") for column in cleaned.columns)
    assert report["cleaner_fit_scope"] == "all_no_split"

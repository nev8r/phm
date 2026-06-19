"""
Test Stage 5 train and eval CLI.
"""

import json
import shutil
import subprocess

import pandas as pd

from tests.infra.dataset_fixtures import create_fake_xjtu_root


def _run_dir(artifact_root, index=0):
    return sorted((artifact_root / "runs").iterdir())[index]


def _write_xjtu_csv(path, scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Horizontal_vibration_signals,Vertical_vibration_signals"]
    for i in range(32):
        lines.append(f"{scale * 0.1 * i},{scale * 0.2 * i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_sequence_samples(root):
    _write_xjtu_csv(root / "35Hz12kN" / "Bearing1_4" / "2.csv", 8.0)
    _write_xjtu_csv(root / "35Hz12kN" / "Bearing1_5" / "2.csv", 9.0)


def test_cli_train_rul_tabular_and_eval_checkpoint(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    train = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=train",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=rul_tabular",
            "model=mlp",
            "trainer=debug",
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

    assert train.returncode == 0, train.stderr
    run_dir = _run_dir(artifact_root)
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    assert checkpoint.exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "metrics" / "val_metrics.json").exists()
    assert (run_dir / "predictions" / "val_predictions.parquet").exists()

    eval_result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=eval",
            f"checkpoint={checkpoint}",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=rul_tabular",
            "model=mlp",
            "trainer=debug",
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

    assert eval_result.returncode == 0, eval_result.stderr
    eval_dir = _run_dir(artifact_root, index=1)
    assert (eval_dir / "metrics" / "test_metrics.json").exists()
    assert (eval_dir / "predictions" / "test_predictions.parquet").exists()


def test_cli_train_health_state_tabular(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=train",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=health_state_tabular",
            "model=mlp",
            "trainer=debug",
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
    metrics = json.loads((run_dir / "metrics" / "val_metrics.json").read_text())
    predictions = pd.read_parquet(run_dir / "predictions" / "val_predictions.parquet")

    assert "Accuracy" in metrics
    assert "y_pred" in predictions.columns


def test_cli_train_rul_sequence_lstm(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    _add_sequence_samples(dataset_root)
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=train",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "task=rul_sequence",
            "task.sequence.length=2",
            "model=lstm",
            "trainer=debug",
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

    assert report["input_mode"] == "feature_sequence"
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "predictions" / "test_predictions.parquet").exists()

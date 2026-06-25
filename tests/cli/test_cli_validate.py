"""
Test the Stage 0 validate CLI.

Purpose: verify test the stage 0 validate cli behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import json
import shutil
import subprocess
import sys

from omegaconf import OmegaConf


def test_cli_validate_creates_stage0_artifacts(tmp_path):
    artifact_root = tmp_path / "artifacts"
    hydra_dir = tmp_path / "hydra"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "USTC.SSE.BearingPrediction.cli.main",
            "--config-name",
            "smoke",
            "mode=validate",
            f"project.artifact_root={artifact_root}",
            f"hydra.run.dir={hydra_dir}",
            "hydra.output_subdir=null",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dirs = sorted((artifact_root / "runs").iterdir())
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]
    resolved = OmegaConf.load(run_dir / "config" / "resolved.yaml")
    report = json.loads((run_dir / "validation_report.json").read_text())
    metadata = json.loads((run_dir / "run.json").read_text())

    assert resolved.mode == "validate"
    assert set(["project", "run", "dataset", "split", "feature", "label", "task", "model", "trainer", "callback"]).issubset(resolved.keys())
    assert report["ok"] is True
    assert report["mode"] == "validate"
    assert report["dataset"] == "XJTU-SY"
    assert metadata["run_dir"] == str(run_dir)


def test_bp_console_script_uses_root_conf_directory(tmp_path):
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    assert bp is not None
    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=validate",
            f"project.artifact_root={artifact_root}",
            "hydra.output_subdir=null",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dirs = sorted((artifact_root / "runs").iterdir())
    assert len(run_dirs) == 1

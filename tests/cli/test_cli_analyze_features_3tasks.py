"""
Smoke tests for the three-task feature analysis configuration.

Purpose: verify smoke tests for the three-task feature analysis configuration behavior
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


EXPECTED_LABEL_COLUMNS = {
    "piecewise_rul_norm",
    "health_state_id",
    "health_state_name",
    "early_fault",
}
FORBIDDEN_FAULT_TYPE_COLUMNS = {
    "fault_type_stage_id",
    "fault_type_stage_name",
}


def _run_dir(artifact_root):
    return sorted((artifact_root / "runs").iterdir())[0]


def _run_analyze_features(args):
    bp = shutil.which("bp")
    result = subprocess.run(
        [bp, "--config-name", "smoke", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _assert_three_task_outputs(run_dir):
    report = json.loads((run_dir / "analysis" / "analysis_report.json").read_text())
    labels = pd.read_parquet(run_dir / "labels" / "labels.parquet")

    assert report["analysis_name"] == "full_feature_analysis_3tasks"
    assert report["ok"] is True
    assert EXPECTED_LABEL_COLUMNS.issubset(labels.columns)
    assert FORBIDDEN_FAULT_TYPE_COLUMNS.isdisjoint(labels.columns)
    for name in [
        "analysis/feature_ranking.csv",
        "analysis/feature_cards.csv",
        "analysis/feature_recommendations.md",
        "labels/labels.parquet",
    ]:
        assert (run_dir / name).exists()


def test_cli_analyze_features_xjtu_three_tasks_without_fault_type(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"

    _run_analyze_features([
        "mode=analyze_features",
        "dataset=xjtu_sy",
        "split=xjtu_bearing_index_split",
        "feature=manual_basic",
        "label=degradation_three_tasks",
        "analysis=full_feature_analysis_3tasks",
        f"dataset.root={dataset_root}",
        f"project.artifact_root={artifact_root}",
        "analysis.plots.enabled=false",
    ])

    _assert_three_task_outputs(_run_dir(artifact_root))


def test_cli_analyze_features_phm2012_three_tasks_without_fault_type(tmp_path):
    dataset_root = create_fake_phm2012_root(tmp_path / "phm2012")
    artifact_root = tmp_path / "artifacts"

    _run_analyze_features([
        "mode=analyze_features",
        "dataset=phm2012",
        "split=phm2012_official",
        "feature=manual_basic",
        "label=degradation_three_tasks",
        "analysis=full_feature_analysis_3tasks",
        f"dataset.root={dataset_root}",
        f"project.artifact_root={artifact_root}",
        "analysis.plots.enabled=false",
    ])

    _assert_three_task_outputs(_run_dir(artifact_root))

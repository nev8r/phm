"""
Test Stage 6 analyze_features CLI.
"""

import json
import shutil
import subprocess

from tests.infra.dataset_fixtures import create_fake_phm2012_root, create_fake_xjtu_root


def _run_dir(artifact_root):
    return sorted((artifact_root / "runs").iterdir())[0]


def test_cli_analyze_features_xjtu_full_analysis(tmp_path):
    dataset_root = create_fake_xjtu_root(tmp_path / "xjtu")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=analyze_features",
            "dataset=xjtu_sy",
            "split=xjtu_leave_one_bearing_out",
            "feature=manual_basic",
            "label=degradation_basic",
            "analysis=full_feature_analysis",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
            "split.condition_id=35Hz12kN",
            "split.test_bearing_id=Bearing1_5",
            "split.val_bearing_id=Bearing1_4",
            "analysis.plots.top_k=2",
            "analysis.plots.max_bearings=2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = _run_dir(artifact_root)
    report = json.loads((run_dir / "analysis" / "analysis_report.json").read_text())

    for name in [
        "analysis_spec.json",
        "analysis_report.json",
        "feature_summary.csv",
        "rul_correlation.csv",
        "degradation_scores.csv",
        "health_state_separability.csv",
        "early_fault_scores.csv",
        "feature_ranking.csv",
        "feature_cards.csv",
        "feature_recommendations.md",
        "leakage_report.json",
        "figures/rul_top_features.png",
        "figures/health_state_boxplots.png",
        "figures/early_fault_effects.png",
        "figures/feature_recommendation_matrix.png",
    ]:
        assert (run_dir / "analysis" / name).exists()
    assert any((run_dir / "analysis" / "figures" / "curves").glob("*.png"))
    assert report["ok"] is True


def test_cli_analyze_features_phm2012_skips_fault_type(tmp_path):
    dataset_root = create_fake_phm2012_root(tmp_path / "phm2012")
    artifact_root = tmp_path / "artifacts"
    bp = shutil.which("bp")

    result = subprocess.run(
        [
            bp,
            "--config-name",
            "smoke",
            "mode=analyze_features",
            "dataset=phm2012",
            "split=none",
            "feature=manual_basic",
            "label=degradation_basic",
            "analysis=full_feature_analysis",
            f"dataset.root={dataset_root}",
            f"project.artifact_root={artifact_root}",
            "analysis.plots.enabled=false",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = _run_dir(artifact_root)
    report = json.loads((run_dir / "analysis" / "analysis_report.json").read_text())

    assert report["fault_type_skipped"] is True
    assert (run_dir / "analysis" / "feature_ranking.csv").exists()

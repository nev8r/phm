"""
Metric-driven Issue 2 evidence tests

this file is for validating tsfresh/sktime baseline evidence and strict SOTA status gates

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

from USTC.SSE.BearingPrediction.experiments.sota import (
    SotaReproductionRecord,
    SotaTargetRecord,
    validate_reproduction_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "reproduction-evidence"


def test_issue2_required_scripts_and_optional_dependencies_are_declared() -> None:
    expected_scripts = [
        PROJECT_ROOT / "scripts" / "run_tsfresh_feature_analysis.py",
        PROJECT_ROOT / "scripts" / "run_tsfresh_rul_baseline.py",
        PROJECT_ROOT / "scripts" / "run_sktime_rul_baseline.py",
        PROJECT_ROOT / "scripts" / "build_strict_repeated_seed_summary.py",
    ]
    for script_path in expected_scripts:
        assert script_path.exists(), f"missing Issue 2 script: {script_path.relative_to(PROJECT_ROOT)}"

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    advanced_dependencies = pyproject["project"]["optional-dependencies"]["advanced"]
    assert any(str(dependency).startswith("tsfresh") for dependency in advanced_dependencies)
    assert any(str(dependency).startswith("sktime") for dependency in advanced_dependencies)


def test_tsfresh_feature_relevance_artifact_has_train_only_selection_schema() -> None:
    relevance_path = EVIDENCE_DIR / "tsfresh_feature_relevance_summary.csv"
    markdown_path = EVIDENCE_DIR / "tsfresh_feature_relevance_summary.md"
    assert relevance_path.exists()
    assert markdown_path.exists()

    relevance = pd.read_csv(relevance_path)
    required_columns = {
        "feature_name",
        "dataset_name",
        "condition_name",
        "target_name",
        "score",
        "p_value",
        "correlation",
        "selected",
        "feature_group",
        "interpretation",
        "selection_split",
        "selection_scope",
        "feature_set_config",
        "feature_grain",
        "train_bearings",
        "held_out_bearing",
        "overlaps_manual_19",
    }
    assert required_columns.issubset(relevance.columns)
    assert len(relevance) >= 10
    assert set(relevance["selection_split"]) == {"train_only"}
    assert set(relevance["selection_scope"]) == {"train_bearings_only"}
    assert {"MinimalFCParameters", "EfficientFCParameters"}.issubset(set(relevance["feature_set_config"]))
    assert set(relevance["feature_grain"]) == {"single_snapshot"}
    assert set(relevance["held_out_bearing"]) == {"Bearing1_3"}
    assert relevance["train_bearings"].astype(str).str.contains("Bearing1_1").all()
    assert relevance["selected"].astype(bool).any()
    assert relevance["feature_name"].nunique() > 2
    assert set(relevance["target_name"]) == {"rul"}
    assert set(relevance["dataset_name"]) == {"XJTU-SY"}
    assert set(relevance["condition_name"]) == {"condition_1_35Hz12kN"}
    assert relevance["interpretation"].astype(str).str.len().min() > 10

    markdown = markdown_path.read_text(encoding="utf-8")
    for required_text in [
        "MinimalFCParameters",
        "EfficientFCParameters",
        "相关性整体偏弱",
        "train-only",
        "Bearing1_3",
    ]:
        assert required_text in markdown

    for figure_name in [
        "tsfresh_feature_correlation_bar.png",
        "tsfresh_feature_group_distribution.png",
        "tsfresh_top_feature_rul_trend.png",
    ]:
        figure_path = EVIDENCE_DIR / figure_name
        assert figure_path.exists(), f"missing tsfresh feature figure: {figure_name}"
        assert figure_path.stat().st_size > 0


def test_tsfresh_rul_baseline_outputs_repeated_manual_and_selected_feature_results() -> None:
    summary_path = EVIDENCE_DIR / "tsfresh_rul_baseline_summary.csv"
    prediction_path = EVIDENCE_DIR / "tsfresh_rul_baseline_predictions.csv"
    assert summary_path.exists()
    assert prediction_path.exists()

    summary = pd.read_csv(summary_path)
    required_summary_columns = {
        "experiment_name",
        "feature_backend",
        "feature_input",
        "feature_set_config",
        "model_name",
        "dataset_name",
        "condition_name",
        "split_name",
        "seed",
        "run_count",
        "rmse",
        "normalized_rmse",
        "mae",
        "r2",
        "huang_rul_score",
        "phm2012_score",
        "rmse_mean",
        "rmse_std",
        "normalized_rmse_mean",
        "normalized_rmse_std",
        "mae_mean",
        "mae_std",
        "prediction_count",
        "feature_count",
        "selection_split",
        "selection_scope",
        "status",
    }
    assert required_summary_columns.issubset(summary.columns)
    assert {"manual_19", "tsfresh_selected", "manual_19_plus_tsfresh_selected"}.issubset(
        set(summary["feature_input"])
    )
    assert {"MinimalFCParameters", "EfficientFCParameters"}.issubset(set(summary["feature_set_config"]))
    assert (
        summary.loc[summary["feature_input"] == "manual_19", "feature_set_config"].eq("manual_19").all()
    )
    assert set(summary["selection_split"]) == {"train_only"}
    assert set(summary["selection_scope"]) == {"train_bearings_only"}
    assert (summary.groupby(["feature_backend", "feature_set_config"])["seed"].nunique() >= 3).all()
    assert (summary["run_count"] >= 3).all()
    assert (summary["prediction_count"] > 0).all()
    assert set(summary["split_name"]) == {"train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3"}
    assert not summary["status"].astype(str).str.startswith("COMPLETED_WITHOUT_RUN").any()
    assert (
        summary.loc[summary["feature_input"] == "manual_19_plus_tsfresh_selected", "feature_count"]
        > summary.loc[summary["feature_input"] == "manual_19", "feature_count"].max()
    ).all()

    predictions = pd.read_csv(prediction_path)
    required_prediction_columns = {
        "experiment_name",
        "feature_backend",
        "feature_input",
        "feature_set_config",
        "seed",
        "bearing_id",
        "snapshot_index",
        "true_rul",
        "predicted_rul",
        "split_name",
    }
    assert required_prediction_columns.issubset(predictions.columns)
    assert {"manual_19", "tsfresh_selected", "manual_19_plus_tsfresh_selected"}.issubset(
        set(predictions["feature_input"])
    )
    assert (predictions["true_rul"] >= 0).all()
    assert predictions["predicted_rul"].notna().all()

    figure_path = EVIDENCE_DIR / "tsfresh_rul_baseline_nrmse_comparison.png"
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_sktime_baseline_outputs_two_repeated_wrapper_routes() -> None:
    summary_path = EVIDENCE_DIR / "sktime_rul_baseline_summary.csv"
    prediction_path = EVIDENCE_DIR / "sktime_rul_baseline_predictions.csv"
    assert summary_path.exists()
    assert prediction_path.exists()

    summary = pd.read_csv(summary_path)
    required_columns = {
        "experiment_name",
        "baseline_route",
        "model_name",
        "input_format",
        "dataset_name",
        "condition_name",
        "split_name",
        "seed",
        "run_count",
        "rmse",
        "normalized_rmse",
        "mae",
        "r2",
        "huang_rul_score",
        "phm2012_score",
        "prediction_count",
        "panel_instance_count",
        "series_length",
        "status",
    }
    assert required_columns.issubset(summary.columns)
    assert {"rocket_regressor", "time_series_forest_regressor"}.issubset(set(summary["baseline_route"]))
    assert (summary.groupby("baseline_route")["seed"].nunique() >= 3).all()
    assert (summary["run_count"] >= 3).all()
    assert set(summary["input_format"]).issubset({"sktime_3d_panel_numpy"})
    assert set(summary["split_name"]) == {"train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3"}

    predictions = pd.read_csv(prediction_path)
    assert {"baseline_route", "seed", "bearing_id", "snapshot_index", "true_rul", "predicted_rul"}.issubset(
        predictions.columns
    )
    assert set(predictions["baseline_route"]) == {"rocket_regressor", "time_series_forest_regressor"}
    assert predictions["predicted_rul"].notna().all()


def test_strict_repeated_seed_summary_uses_same_config_rows() -> None:
    summary_path = EVIDENCE_DIR / "strict_repeated_seed_summary.csv"
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    required_columns = {
        "model_name",
        "dataset_name",
        "condition_name",
        "split_name",
        "seed",
        "rmse",
        "normalized_rmse",
        "mae",
        "r2",
        "score",
        "epoch",
        "config_hash",
        "config_path",
        "run_count",
        "status",
    }
    assert required_columns.issubset(summary.columns)
    assert {"XLSTM-Transformer", "Feature-Transformer"}.issubset(set(summary["model_name"]))
    assert (summary.groupby("model_name")["seed"].nunique() >= 3).all()
    assert (summary.groupby("model_name")["config_hash"].nunique() == 1).all()
    assert summary["config_hash"].astype(str).str.len().min() >= 12
    assert set(summary["split_name"]) == {"train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3"}
    assert summary["config_path"].astype(str).str.endswith(".json").all()
    for config_path in summary["config_path"].unique():
        assert (PROJECT_ROOT / str(config_path)).exists()


def test_external_sota_blocked_status_has_reproducible_attempt_log() -> None:
    attempts_path = EVIDENCE_DIR / "external_sota_attempts.csv"
    assert attempts_path.exists()
    attempts = pd.read_csv(attempts_path)
    required_columns = {
        "target_id",
        "route_name",
        "attempt_status",
        "attempt_type",
        "command",
        "source_pin_command",
        "source_pin_exit_code",
        "environment_probe_command",
        "environment_probe_exit_code",
        "log_path",
        "failure_reason",
        "next_step",
    }
    assert required_columns.issubset(attempts.columns)
    assert not attempts.empty
    assert {"autorul-pronostia-femto-rmse"}.issubset(set(attempts["target_id"]))
    assert (attempts["source_pin_exit_code"] == 0).all()
    assert (attempts["environment_probe_exit_code"] != 0).all()

    for log_path in attempts["log_path"]:
        resolved = PROJECT_ROOT / str(log_path)
        assert resolved.exists(), f"missing external SOTA attempt log: {log_path}"
        log_text = resolved.read_text(encoding="utf-8")
        assert "environment_probe_exit_code:" in log_text
        assert "failure_reason:" in log_text

    reproduction_path = EVIDENCE_DIR / "open_source_sota_reproduction_summary.csv"
    reproduction = pd.read_csv(reproduction_path)
    blocked_external = reproduction.loc[reproduction["status"].astype(str).str.contains("EXTERNAL_ENV", na=False)]
    assert not blocked_external.empty
    assert (blocked_external["evidence_path"] != "not_available").all()


def test_attempted_external_env_rows_are_valid_non_run_evidence() -> None:
    target = SotaTargetRecord(
        target_id="autorul-pronostia-femto-rmse",
        method_name="AutoRUL",
        dataset_name="PRONOSTIA",
        condition_name="femto_bearing",
        metric_name="rmse",
        target_value=22.52,
        metric_direction="lower",
        source_type="open_source_repo_with_paper",
        source_url="https://github.com/Ennosigaeon/auto-sktime",
        source_commit="fe277d21104be8d2e4bd34db7ed995547007e55b",
        split_description="external AutoRUL split",
        reproducibility_status="open_source_external_env_required",
        license_name="MIT",
        run_command="python remaining_useful_lifetime.py femto_bearing",
    )
    row = SotaReproductionRecord.from_target(
        target,
        experiment_name="AutoRUL-source-probe",
        local_method_name="not_run_in_project_environment",
        local_value=float("nan"),
        local_mean=float("nan"),
        local_std=float("nan"),
        run_count=0,
        seeds="not_run",
        prediction_count=0,
        evidence_path="docs/reproduction-evidence/external_sota_attempts/autorul.log",
        status="ATTEMPTED_EXTERNAL_ENV_BLOCKED",
        notes="attempt log records dependency boundary",
    )

    validate_reproduction_frame(pd.DataFrame([row.to_dict()]), min_run_count=1)

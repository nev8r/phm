"""
Open-source SOTA evidence tests

this file is for validating SOTA target and reproduction evidence

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from USTC.SSE.BearingPrediction.experiments.sota import (
    SotaEvidenceBuilder,
    SotaReproductionRecord,
    SotaTargetRecord,
    calculate_gap_percent,
    validate_reproduction_frame,
    validate_target_frame,
)


def test_gap_percent_treats_lower_error_as_better() -> None:
    gap = calculate_gap_percent(local_value=0.064558, target_value=0.0583, higher_is_better=False)

    assert round(gap, 3) == 10.734


def test_gap_percent_treats_higher_r2_as_better() -> None:
    gap = calculate_gap_percent(local_value=0.950555, target_value=0.9691, higher_is_better=True)

    assert round(gap, 3) == 1.914


def test_target_validator_rejects_targets_without_reproducible_source() -> None:
    target_frame = pd.DataFrame(
        [
            {
                "target_id": "paper-only",
                "method_name": "Paper Only",
                "dataset_name": "PHM2012",
                "condition_name": "condition_1",
                "metric_name": "rmse",
                "target_value": 0.0778,
                "metric_direction": "lower",
                "source_type": "paper_reference_only",
                "source_url": "",
                "source_commit": "",
                "split_description": "official split",
                "reproducibility_status": "reference_only",
                "license_name": "paper",
                "run_command": "not available",
                "notes": "missing source url must fail validation",
            }
        ]
    )

    with pytest.raises(ValueError, match="source_url"):
        validate_target_frame(target_frame)


def test_reproduction_validator_requires_repeated_runs() -> None:
    target = SotaTargetRecord(
        target_id="jiang-xjtu-c1-xlstm-rmse",
        method_name="xLSTM-Transformer",
        dataset_name="XJTU-SY",
        condition_name="condition_1_35Hz12kN",
        metric_name="normalized_rmse",
        target_value=0.0583,
        metric_direction="lower",
        source_type="paper_reference_with_local_reimplementation",
        source_url="https://www.mdpi.com/1424-8220/26/5/1578",
        source_commit="local-implementation",
        split_description="Bearing1_1,1_2,1_4,1_5 train; Bearing1_3 test",
        reproducibility_status="local_reimplementation",
        license_name="paper",
        run_command="uv run python scripts/run_formal_paper_reproductions.py",
        notes="paper target used for gap calculation",
    )
    reproduction = SotaReproductionRecord.from_target(
        target,
        experiment_name="single-seed",
        local_method_name="XLSTM-Transformer",
        local_value=0.064558,
        local_mean=0.064558,
        local_std=0.0,
        run_count=1,
        seeds="2026",
        prediction_count=87,
        evidence_path="tmp/formal_paper_reproductions_50ep/paper_xlstm_transformer/comparison_metrics.csv",
        status="PASS",
        notes="single seed must be rejected by no-cherry-pick validator",
    )

    with pytest.raises(ValueError, match="run_count"):
        validate_reproduction_frame(pd.DataFrame([reproduction.to_dict()]), min_run_count=3)


def test_default_targets_include_autorul_open_source_route() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target_frame = SotaEvidenceBuilder(project_root).default_targets()

    autorul = target_frame.loc[target_frame["target_id"] == "autorul-pronostia-femto-rmse"].iloc[0]
    assert autorul["source_url"] == "https://github.com/Ennosigaeon/auto-sktime"
    assert autorul["source_commit"] == "fe277d21104be8d2e4bd34db7ed995547007e55b"
    assert autorul["license_name"] == "MIT"
    assert "remaining_useful_lifetime.py femto_bearing" in autorul["run_command"]


def test_evidence_builder_maps_repeated_formal_results_to_sota_gap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    builder = SotaEvidenceBuilder(project_root)

    targets = builder.default_targets()
    reproduction_frame = builder.build_reproduction_summary(targets)

    xjtu_feature_transformer = reproduction_frame.loc[
        reproduction_frame["target_id"] == "jiang-xjtu-c1-feature-transformer-rmse"
    ].iloc[0]
    assert xjtu_feature_transformer["local_method_name"] == "Feature-Transformer"
    assert xjtu_feature_transformer["run_count"] >= 3
    assert xjtu_feature_transformer["gap_percent"] <= 25.0
    assert xjtu_feature_transformer["status"] == "PASS"

    xjtu_xlstm = reproduction_frame.loc[
        reproduction_frame["target_id"] == "jiang-xjtu-c1-xlstm-rmse"
    ].iloc[0]
    assert xjtu_xlstm["run_count"] >= 3
    assert xjtu_xlstm["status"] == "NEEDS_OPTIMIZATION"


def test_evidence_builder_includes_rulsurv_rsf_port_reproduction() -> None:
    project_root = Path(__file__).resolve().parents[1]
    builder = SotaEvidenceBuilder(project_root)

    targets = builder.default_targets()
    reproduction_frame = builder.build_reproduction_summary(targets)

    rulsurv_rows = reproduction_frame.loc[
        reproduction_frame["target_id"] == "rulsurv-xjtu-high-rsf-true-mae"
    ]
    assert not rulsurv_rows.empty

    original_protocol = rulsurv_rows.loc[
        rulsurv_rows["experiment_name"] == "RULSurv-RSF-port-rulsurv_original_25pct_censored_cv"
    ].iloc[0]
    assert original_protocol["local_method_name"] == "RULSurv RSF port"
    assert original_protocol["source_commit"] == "6365e0832de9724a5bcbbac4557c6643dfb78d91"
    assert original_protocol["run_count"] >= 3
    assert original_protocol["prediction_count"] == 611
    assert original_protocol["local_mean"] <= original_protocol["target_value"]
    assert original_protocol["status"] == "PROTOCOL_PASS"

    project_holdout = rulsurv_rows.loc[
        rulsurv_rows["experiment_name"] == "RULSurv-RSF-port-project_bearing1_3_holdout_migration"
    ].iloc[0]
    assert project_holdout["run_count"] >= 3
    assert project_holdout["prediction_count"] == 157
    assert project_holdout["local_mean"] <= 15.75
    assert project_holdout["status"] == "MIGRATION_PASS"

    blocked_rows = reproduction_frame.loc[reproduction_frame["status"].astype(str).str.startswith("BLOCKED")]
    assert (blocked_rows["run_count"] == 0).all()

    rgpd_reference = reproduction_frame.loc[
        reproduction_frame["target_id"] == "rgpd-phm2012-reference-rmse"
    ].iloc[0]
    assert rgpd_reference["status"] == "REFERENCE_ONLY"
    assert rgpd_reference["run_count"] == 0

    metric_summary = builder.build_metric_driven_summary(reproduction_frame)
    non_run_metric_rows = metric_summary.loc[
        metric_summary["status"].astype(str).str.startswith("BLOCKED")
        | (metric_summary["status"] == "REFERENCE_ONLY")
    ]
    assert (non_run_metric_rows["feature_backend"] == "not_run").all()


def test_rulsurv_snapshot_audit_proves_full_condition_usage() -> None:
    project_root = Path(__file__).resolve().parents[1]
    audit_path = project_root / "docs" / "reproduction-evidence" / "rulsurv_rsf_port" / "rulsurv_rsf_port_snapshot_audit.csv"
    audit_frame = pd.read_csv(audit_path)

    expected_counts = {
        "Bearing1_1": (123, 122),
        "Bearing1_2": (161, 160),
        "Bearing1_3": (158, 157),
        "Bearing1_4": (122, 121),
        "Bearing1_5": (52, 51),
        "TOTAL": (616, 611),
    }
    for bearing_id, (raw_count, used_count) in expected_counts.items():
        row = audit_frame.loc[audit_frame["bearing_id"] == bearing_id].iloc[0]
        assert row["raw_snapshot_count"] == raw_count
        assert row["used_snapshot_count"] == used_count
        assert row["uses_sampling_cap"] in {False, "False"}

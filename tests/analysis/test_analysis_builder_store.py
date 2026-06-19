"""
Test Stage 6 analysis builder and store.
"""

import json

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.analysis.AnalysisBuilder import AnalysisBuilder
from USTC.SSE.BearingPrediction.analysis.AnalysisStore import AnalysisStore
from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult


def _features():
    return pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(6)],
        "dataset": ["XJTU-SY"] * 6,
        "bearing_id": ["Bearing1_1"] * 3 + ["Bearing1_2"] * 3,
        "condition_id": ["35Hz12kN"] * 6,
        "source_group": [None] * 6,
        "sample_id": ["000000", "000001", "000002"] * 2,
        "timestep": [0, 1, 2] * 2,
        "feature_good": [0, 1, 2, 0, 1, 2],
        "feature_bad": [0.2, 0.1, 0.3, 0.6, 0.5, 0.4],
    })


def _labels():
    labels = _features()[["sample_uid", "dataset", "bearing_id", "condition_id", "source_group", "sample_id", "timestep"]].copy()
    labels["piecewise_rul_norm"] = [1, 0.5, 0, 1, 0.5, 0]
    labels["health_state_id"] = [0, 1, 3, 0, 1, 3]
    labels["health_state_name"] = ["healthy", "slight", "severe"] * 2
    labels["early_fault"] = [0, 1, 1, 0, 1, 1]
    return labels


def test_analysis_builder_creates_outputs_and_skips_missing_fault_type():
    cfg = OmegaConf.create({
        "name": "full_feature_analysis",
        "version": "v1",
        "feature_source": "raw",
        "scope": {"fit_scope": "train_only", "report_splits": ["all"]},
        "summary": {"enabled": True},
        "rul_correlation": {"enabled": True, "target_column": "piecewise_rul_norm", "methods": ["spearman"]},
        "degradation_scores": {"enabled": True, "interpolation_points": 10},
        "health_state": {"enabled": True, "target_column": "health_state_id"},
        "early_fault": {"enabled": True, "target_column": "early_fault"},
        "fault_type": {"enabled": True, "target_column": "fault_type_stage_id", "skip_if_missing": True},
        "ranking": {"enabled": True},
        "leakage": {"enabled": True},
        "plots": {"enabled": False},
    })

    outputs = AnalysisBuilder(cfg).build(_features(), _labels(), index=_features(), split_result=None, hi=None, fpt=None)

    assert outputs["analysis_spec"]["hash"]
    assert outputs["analysis_report"]["ok"] is True
    assert outputs["analysis_report"]["fault_type_skipped"] is True
    assert not outputs["feature_ranking"].empty


def test_analysis_builder_does_not_treat_internal_split_column_as_feature():
    cfg = OmegaConf.create({
        "name": "full_feature_analysis",
        "version": "v1",
        "feature_source": "raw",
        "scope": {"fit_scope": "train_only", "report_splits": ["train"]},
        "summary": {"enabled": True},
        "rul_correlation": {"enabled": True, "target_column": "piecewise_rul_norm"},
        "degradation_scores": {"enabled": True, "interpolation_points": 10},
        "health_state": {"enabled": True, "target_column": "health_state_id"},
        "early_fault": {"enabled": True, "target_column": "early_fault"},
        "ranking": {"enabled": True},
        "leakage": {"enabled": True},
        "plots": {"enabled": False},
    })
    split = SplitResult(
        name="unit_split",
        train_sample_uids=["s0", "s1", "s2"],
        val_sample_uids=["s3"],
        test_sample_uids=["s4", "s5"],
        train_bearings=["Bearing1_1"],
        val_bearings=["Bearing1_2"],
        test_bearings=["Bearing1_2"],
    )

    outputs = AnalysisBuilder(cfg).build(_features(), _labels(), index=_features(), split_result=split)

    assert "__split" not in set(outputs["rul_correlation"]["feature"])
    assert "__split" not in set(outputs["degradation_scores"]["feature"])
    assert "__split" not in set(outputs["feature_ranking"]["feature"])


def test_analysis_store_writes_artifacts(tmp_path):
    cfg = OmegaConf.create({
        "name": "full_feature_analysis",
        "version": "v1",
        "feature_source": "raw",
        "scope": {"fit_scope": "all_no_split", "report_splits": ["all"]},
        "summary": {"enabled": True},
        "rul_correlation": {"enabled": True, "target_column": "piecewise_rul_norm"},
        "degradation_scores": {"enabled": True, "interpolation_points": 10},
        "health_state": {"enabled": True, "target_column": "health_state_id"},
        "early_fault": {"enabled": True, "target_column": "early_fault"},
        "fault_type": {"enabled": True, "target_column": "fault_type_stage_id", "skip_if_missing": True},
        "ranking": {"enabled": True},
        "leakage": {"enabled": True},
        "plots": {"enabled": False},
    })
    outputs = AnalysisBuilder(cfg).build(_features(), _labels(), index=_features(), split_result=None, hi=None, fpt=None)

    AnalysisStore(ArtifactManager(tmp_path), write_csv=True, write_figures=False).save(outputs)

    assert (tmp_path / "analysis" / "analysis_spec.json").exists()
    assert (tmp_path / "analysis" / "analysis_report.json").exists()
    assert (tmp_path / "analysis" / "feature_summary.csv").exists()
    assert (tmp_path / "analysis" / "feature_ranking.parquet").exists()
    assert json.loads((tmp_path / "analysis" / "analysis_report.json").read_text())["ok"] is True

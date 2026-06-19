"""
Test Stage 6 feature analyzers.
"""

import pandas as pd
from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.analysis.DegradationScoreAnalyzer import DegradationScoreAnalyzer
from USTC.SSE.BearingPrediction.analysis.EarlyFaultFeatureAnalyzer import EarlyFaultFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis.FeatureRanking import build_feature_ranking
from USTC.SSE.BearingPrediction.analysis.FeatureSummaryAnalyzer import FeatureSummaryAnalyzer
from USTC.SSE.BearingPrediction.analysis.HealthStateFeatureAnalyzer import HealthStateFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis.LeakageGuard import LeakageGuard
from USTC.SSE.BearingPrediction.analysis.RulFeatureAnalyzer import RulFeatureAnalyzer


def _features():
    return pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(8)],
        "dataset": ["XJTU-SY"] * 8,
        "bearing_id": ["Bearing1_1"] * 4 + ["Bearing1_2"] * 4,
        "condition_id": ["35Hz12kN"] * 8,
        "source_group": [None] * 8,
        "sample_id": [f"{i:06d}" for i in range(4)] * 2,
        "timestep": list(range(4)) * 2,
        "feature_good": [0, 1, 2, 3, 0, 1, 2, 3],
        "feature_bad": [0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7],
        "feature_constant": [1.0] * 8,
    })


def _labels():
    return pd.DataFrame({
        "sample_uid": [f"s{i}" for i in range(8)],
        "dataset": ["XJTU-SY"] * 8,
        "bearing_id": ["Bearing1_1"] * 4 + ["Bearing1_2"] * 4,
        "condition_id": ["35Hz12kN"] * 8,
        "source_group": [None] * 8,
        "sample_id": [f"{i:06d}" for i in range(4)] * 2,
        "timestep": list(range(4)) * 2,
        "piecewise_rul_norm": [1, 0.66, 0.33, 0, 1, 0.66, 0.33, 0],
        "health_state_id": [0, 1, 2, 3, 0, 1, 2, 3],
        "health_state_name": ["healthy", "slight", "moderate", "severe"] * 2,
        "early_fault": [0, 0, 1, 1, 0, 0, 1, 1],
        "fault_type_stage_id": [0, 1, 1, 2, 0, 1, 1, 2],
        "fault_type_stage_name": ["normal", "degraded_unknown", "degraded_unknown", "outer"] * 2,
    })


def test_feature_summary_reports_missing_variance_and_constant_status():
    summary = FeatureSummaryAnalyzer().analyze(_features(), split_result=None)

    constant = summary[(summary["feature"] == "feature_constant") & (summary["split"] == "all")].iloc[0]
    assert constant["is_constant"] is True or bool(constant["is_constant"]) is True
    assert "variance" in summary.columns
    assert "missing_count" in summary.columns


def test_rul_feature_analyzer_ranks_monotonic_feature_higher_than_noise():
    result = RulFeatureAnalyzer(OmegaConf.create({"target_column": "piecewise_rul_norm"})).analyze(_features(), _labels())
    good = result[result["feature"] == "feature_good"].iloc[0]
    bad = result[result["feature"] == "feature_bad"].iloc[0]

    assert good["abs_spearman"] > 0.99
    assert good["abs_spearman"] > bad["abs_spearman"]


def test_degradation_score_analyzer_prefers_monotonic_feature():
    result = DegradationScoreAnalyzer(OmegaConf.create({"interpolation_points": 10})).analyze(_features())
    good = result[result["feature"] == "feature_good"].iloc[0]
    bad = result[result["feature"] == "feature_bad"].iloc[0]

    assert good["monotonicity"] > bad["monotonicity"]


def test_health_state_analyzer_scores_separable_feature():
    result = HealthStateFeatureAnalyzer(OmegaConf.create({"target_column": "health_state_id"})).analyze(_features(), _labels())
    good = result[result["feature"] == "feature_good"].iloc[0]
    bad = result[result["feature"] == "feature_bad"].iloc[0]

    assert good["fisher_score"] > bad["fisher_score"]
    assert good["class_count"] == 4
    assert "state_0_mean" in result.columns
    assert "state_3_std" in result.columns


def test_early_fault_analyzer_scores_sensitive_feature():
    result = EarlyFaultFeatureAnalyzer(OmegaConf.create({"target_column": "early_fault"})).analyze(_features(), _labels())
    good = result[result["feature"] == "feature_good"].iloc[0]

    assert good["cohens_d"] > 0
    assert good["auc"] > 0.8
    assert "auc_abs" in result.columns
    assert "healthy_std" in result.columns
    assert "fault_std" in result.columns


def test_leakage_guard_marks_hi_source_feature():
    hi = pd.DataFrame({"sample_uid": ["s0"], "hi_source_column": ["feature_good"]})
    report = LeakageGuard(OmegaConf.create({"fit_scope": "train_only"})).check(_features(), hi=hi, fpt=None)

    assert report["ok"] is True
    assert report["warnings"][0]["feature"] == "feature_good"


def test_feature_ranking_contains_required_columns_and_recommendations():
    rul = RulFeatureAnalyzer(OmegaConf.create({"target_column": "piecewise_rul_norm"})).analyze(_features(), _labels())
    degradation = DegradationScoreAnalyzer(OmegaConf.create({"interpolation_points": 10})).analyze(_features())
    health = HealthStateFeatureAnalyzer(OmegaConf.create({"target_column": "health_state_id"})).analyze(_features(), _labels())
    early = EarlyFaultFeatureAnalyzer(OmegaConf.create({"target_column": "early_fault"})).analyze(_features(), _labels())
    ranking = build_feature_ranking(
        feature_columns=["feature_good", "feature_bad", "feature_constant"],
        rul_correlation=rul,
        degradation_scores=degradation,
        health_state_separability=health,
        early_fault_scores=early,
        fault_type_scores=None,
        label_source_features={"feature_good"},
    )

    for column in ["rul_score", "health_score", "early_fault_score", "fault_type_score", "overall_score", "rank_fault_type", "recommended_for", "label_source_warning"]:
        assert column in ranking.columns
    assert ranking.iloc[0]["rank_overall"] == 1
    assert ranking[ranking["feature"] == "feature_good"].iloc[0]["is_label_source"] is True or bool(ranking[ranking["feature"] == "feature_good"].iloc[0]["is_label_source"]) is True

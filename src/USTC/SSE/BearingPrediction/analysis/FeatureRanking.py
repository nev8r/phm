"""
Feature ranking utilities.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

import math
from typing import Optional, Set

import pandas as pd

from USTC.SSE.BearingPrediction.analysis._helpers import minmax


def build_feature_ranking(
        feature_columns,
        rul_correlation: Optional[pd.DataFrame] = None,
        degradation_scores: Optional[pd.DataFrame] = None,
        health_state_separability: Optional[pd.DataFrame] = None,
        early_fault_scores: Optional[pd.DataFrame] = None,
        fault_type_scores: Optional[pd.DataFrame] = None,
        label_source_features: Optional[Set[str]] = None,
) -> pd.DataFrame:
    label_source_features = label_source_features or set()
    ranking = pd.DataFrame({"feature": list(feature_columns)})
    ranking = _merge(ranking, _rul_scores(rul_correlation, degradation_scores), "rul_score")
    ranking = _merge(ranking, _health_scores(health_state_separability), "health_score")
    ranking = _merge(ranking, _early_scores(early_fault_scores), "early_fault_score")
    ranking = _merge(ranking, _fault_type_scores(fault_type_scores), "fault_type_score")
    for column in ["rul_score", "health_score", "early_fault_score", "fault_type_score"]:
        if column not in ranking.columns:
            ranking[column] = 0.0
        ranking[column] = ranking[column].fillna(0.0)
    ranking["overall_score"] = ranking[["rul_score", "health_score", "early_fault_score", "fault_type_score"]].mean(axis=1)
    ranking["rank_rul"] = _rank(ranking["rul_score"])
    ranking["rank_health"] = _rank(ranking["health_score"])
    ranking["rank_early_fault"] = _rank(ranking["early_fault_score"])
    ranking["rank_fault_type"] = _rank(ranking["fault_type_score"])
    ranking["rank_overall"] = _rank(ranking["overall_score"])
    ranking["is_label_source"] = ranking["feature"].isin(label_source_features)
    ranking["label_source_warning"] = ranking["is_label_source"].map(
        lambda value: "Used as HI source for FPT-based labels; do not overclaim independent detection ability." if value else ""
    )
    limits = _recommendation_limits(ranking)
    ranking["recommended_for"] = ranking.apply(lambda row: _recommended_for(row, limits), axis=1)
    return ranking.sort_values(["rank_overall", "feature"]).reset_index(drop=True)


def _merge(ranking: pd.DataFrame, scores: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if scores is None or scores.empty:
        ranking[score_column] = 0.0
        return ranking
    return ranking.merge(scores[["feature", score_column]], on="feature", how="left")


def _rul_scores(rul: Optional[pd.DataFrame], degradation: Optional[pd.DataFrame]) -> pd.DataFrame:
    if rul is None or rul.empty:
        return pd.DataFrame(columns=["feature", "rul_score"])
    data = rul[["feature", "abs_spearman"]].copy()
    if degradation is not None and not degradation.empty:
        data = data.merge(degradation, on="feature", how="left")
    for column in ["abs_spearman", "monotonicity", "robustness", "trendability"]:
        if column not in data.columns:
            data[column] = 0.0
        data[column] = data[column].fillna(0.0)
    data["rul_score"] = (
        0.35 * data["abs_spearman"]
        + 0.25 * data["monotonicity"]
        + 0.20 * data["robustness"]
        + 0.20 * data["trendability"]
    )
    return data[["feature", "rul_score"]]


def _health_scores(health: Optional[pd.DataFrame]) -> pd.DataFrame:
    if health is None or health.empty:
        return pd.DataFrame(columns=["feature", "health_score"])
    data = health.copy()
    data["health_score"] = (
        0.4 * minmax(_column_or_zero(data, "mutual_information"))
        + 0.4 * minmax(_column_or_zero(data, "fisher_score"))
        + 0.2 * minmax(_column_or_zero(data, "anova_f"))
    )
    return data[["feature", "health_score"]]


def _early_scores(early: Optional[pd.DataFrame]) -> pd.DataFrame:
    if early is None or early.empty:
        return pd.DataFrame(columns=["feature", "early_fault_score"])
    data = early.copy()
    auc_column = "auc_abs" if "auc_abs" in data.columns else "auc"
    data["early_fault_score"] = 0.5 * minmax(data["cohens_d"].abs()) + 0.5 * minmax(data[auc_column])
    return data[["feature", "early_fault_score"]]


def _fault_type_scores(fault: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fault is None or fault.empty:
        return pd.DataFrame(columns=["feature", "fault_type_score"])
    data = fault.copy()
    data["fault_type_score"] = 0.5 * minmax(_column_or_zero(data, "mutual_information")) + 0.5 * minmax(_column_or_zero(data, "fisher_score"))
    return data[["feature", "fault_type_score"]]


def _rank(series: pd.Series) -> pd.Series:
    return series.rank(method="min", ascending=False).astype(int)


def _recommendation_limits(ranking: pd.DataFrame) -> dict:
    return {
        task: _top_limit(ranking[score_column])
        for task, score_column in [
            ("RUL", "rul_score"),
            ("HealthState", "health_score"),
            ("EarlyFault", "early_fault_score"),
            ("FaultType", "fault_type_score"),
        ]
    }


def _top_limit(series: pd.Series) -> int:
    nonzero = int((series > 0).sum())
    if nonzero == 0:
        return 0
    return max(1, int(math.ceil(nonzero * 0.2)))


def _recommended_for(row, limits: dict) -> str:
    recommendations = []
    if row["rul_score"] > 0 and row["rank_rul"] <= limits["RUL"]:
        recommendations.append("RUL")
    if row["health_score"] > 0 and row["rank_health"] <= limits["HealthState"]:
        recommendations.append("HealthState")
    if row["early_fault_score"] > 0 and row["rank_early_fault"] <= limits["EarlyFault"]:
        recommendations.append("EarlyFault")
    if row["fault_type_score"] > 0 and row["rank_fault_type"] <= limits["FaultType"]:
        recommendations.append("FaultType")
    return ",".join(recommendations)


def _column_or_zero(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series([0.0] * len(data), index=data.index)

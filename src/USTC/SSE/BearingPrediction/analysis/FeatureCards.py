"""
Feature recommendation cards.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Iterable, Set

import pandas as pd


LABEL_SOURCE_WARNING = "Used as HI source for FPT-based labels; do not overclaim independent detection ability."


def build_feature_cards(feature_ranking: pd.DataFrame, label_source_features: Iterable[str] = ()) -> pd.DataFrame:
    label_sources: Set[str] = set(label_source_features)
    rows = []
    for _, row in feature_ranking.iterrows():
        feature = str(row["feature"])
        is_label_source = bool(feature in label_sources or row.get("is_label_source", False))
        recommended_for = str(row.get("recommended_for", ""))
        caveat = _caveat(feature)
        if is_label_source:
            caveat = _join_text(caveat, LABEL_SOURCE_WARNING)
        rows.append({
            "feature": feature,
            "feature_family": _feature_family(feature),
            "physical_meaning": _physical_meaning(feature),
            "rul_score": float(row.get("rul_score", 0.0)),
            "health_score": float(row.get("health_score", 0.0)),
            "early_fault_score": float(row.get("early_fault_score", 0.0)),
            "fault_type_score": float(row.get("fault_type_score", 0.0)),
            "overall_score": float(row.get("overall_score", 0.0)),
            "recommended_for": recommended_for,
            "why": _why(row),
            "caveat": caveat,
            "is_label_source": is_label_source,
            "label_source_warning": LABEL_SOURCE_WARNING if is_label_source else "",
            "example_plot": f"figures/curves/{_safe_name(feature)}.png",
        })
    return pd.DataFrame(rows)


def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if "tsfresh" in lower:
        return "tsfresh"
    if "__spectral__" in lower or "frequency" in lower or "entropy" in lower or "centroid" in lower:
        return "spectral"
    if "__time__" in lower or any(token in lower for token in ["rms", "std", "ptp", "kurtosis", "crest", "impulse", "clearance"]):
        return "time_domain"
    return "unknown"


def _physical_meaning(feature: str) -> str:
    lower = feature.lower()
    if "rms" in lower:
        return "RMS vibration energy/amplitude level."
    if "std" in lower or "variance" in lower:
        return "Signal dispersion and vibration intensity."
    if "mean_abs" in lower:
        return "Average absolute vibration magnitude."
    if "ptp" in lower or "peak_to_peak" in lower:
        return "Peak-to-peak vibration amplitude."
    if "kurtosis" in lower:
        return "Impulsive shock sensitivity and tail heaviness."
    if "crest_factor" in lower:
        return "Peak shock relative to RMS level."
    if "impulse_factor" in lower:
        return "Peak shock relative to mean absolute level."
    if "clearance_factor" in lower:
        return "Sharp impulse sensitivity."
    if "centroid" in lower:
        return "Spectral energy center shift."
    if "rms_frequency" in lower:
        return "RMS frequency content shift."
    if "peak_frequency" in lower:
        return "Dominant frequency location."
    if "entropy" in lower:
        return "Spectral complexity or broadband excitation."
    if "abs_energy" in lower or "energy" in lower:
        return "Signal energy over the sample window."
    return "Numerical feature extracted from the vibration snapshot."


def _why(row) -> str:
    reasons = []
    if "RUL" in str(row.get("recommended_for", "")):
        reasons.append("high RUL score from correlation/trend metrics")
    if "HealthState" in str(row.get("recommended_for", "")):
        reasons.append("separates health-state label distributions")
    if "EarlyFault" in str(row.get("recommended_for", "")):
        reasons.append("changes between healthy and post-FPT samples")
    if "FaultType" in str(row.get("recommended_for", "")):
        reasons.append("weakly separates fault-type stage labels")
    if reasons:
        return "; ".join(reasons)
    best_task = _best_task(row)
    if best_task:
        return f"highest relative score is for {best_task}, but it is outside the top recommendation band"
    return "no strong task-specific evidence in the current analysis run"


def _best_task(row) -> str:
    scores = {
        "RUL": float(row.get("rul_score", 0.0)),
        "HealthState": float(row.get("health_score", 0.0)),
        "EarlyFault": float(row.get("early_fault_score", 0.0)),
        "FaultType": float(row.get("fault_type_score", 0.0)),
    }
    task, score = max(scores.items(), key=lambda item: item[1])
    return task if score > 0 else ""


def _caveat(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ["kurtosis", "crest_factor", "impulse_factor", "clearance_factor"]):
        return "Sensitive to spikes/noise; verify curve stability before using as a RUL main feature."
    if "entropy" in lower:
        return "Useful for spectral complexity, but physical interpretation is weaker than energy or bearing-frequency features."
    if "centroid" in lower or "frequency" in lower:
        return "Can be affected by operating-condition changes."
    if "tsfresh" in lower:
        return "Requires train-only scaling and cross-bearing validation before interpretation."
    return "Confirm with train-only ranking and visual plots before overclaiming."


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"

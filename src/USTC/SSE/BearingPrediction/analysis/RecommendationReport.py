"""
Markdown feature recommendation report.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import pandas as pd


def build_feature_recommendations(
        feature_cards: pd.DataFrame,
        feature_ranking: pd.DataFrame,
        leakage_report: Dict,
        analysis_report: Dict,
        top_k: int = 5,
) -> str:
    lines: List[str] = [
        "# Feature Recommendations",
        "",
        "## Summary",
        f"- Analysis: {analysis_report.get('analysis_name', 'analysis')}",
        f"- Feature source: {analysis_report.get('feature_source', 'unknown')}",
        f"- Ranking scope: {analysis_report.get('fit_scope', 'unknown')}",
        f"- Ranked features: {analysis_report.get('num_ranked_features', 0)}",
        f"- Leakage warnings: {analysis_report.get('num_leakage_warnings', 0)}",
        "",
    ]
    for title, task_name, rank_column in [
        ("RUL", "RUL", "rank_rul"),
        ("Health State", "HealthState", "rank_health"),
        ("Early Fault", "EarlyFault", "rank_early_fault"),
        ("Fault Type", "FaultType", "rank_fault_type"),
    ]:
        lines.extend(_task_section(title, task_name, rank_column, feature_cards, feature_ranking, top_k))
    lines.extend(_leakage_section(leakage_report))
    return "\n".join(lines).rstrip() + "\n"


def _task_section(title: str, task_name: str, rank_column: str, cards: pd.DataFrame, ranking: pd.DataFrame, top_k: int) -> List[str]:
    merged = ranking[["feature", rank_column]].merge(cards, on="feature", how="left")
    recommended = merged[merged["recommended_for"].fillna("").str.contains(task_name, regex=False)].copy()
    if recommended.empty:
        recommended = merged[merged[rank_column] <= top_k].copy()
    recommended = recommended.sort_values([rank_column, "feature"]).head(top_k)
    lines = [f"## {title}", "Top features:"]
    if recommended.empty:
        lines.append("- No feature reached the recommendation threshold in this run.")
        lines.append("")
        return lines
    for index, (_, row) in enumerate(recommended.iterrows(), start=1):
        caveat = str(row.get("caveat", "") or "")
        label_warning = str(row.get("label_source_warning", "") or "")
        if label_warning and label_warning not in caveat:
            caveat = f"{caveat} {label_warning}".strip()
        lines.append(f"{index}. {row['feature']}")
        lines.append(f"   - reason: {row.get('why', '')}")
        lines.append(f"   - caveat: {caveat}")
        lines.append(f"   - plot: {row.get('example_plot', '')}")
    lines.append("")
    return lines


def _leakage_section(leakage_report: Dict) -> List[str]:
    lines = ["## Leakage Warnings"]
    warnings = leakage_report.get("warnings", []) if leakage_report else []
    if not warnings:
        lines.extend(["- No leakage warning was raised.", ""])
        return lines
    for warning in warnings:
        feature = warning.get("feature")
        prefix = f"- `{feature}`: " if feature else "- "
        lines.append(f"{prefix}{warning.get('message', '')}")
    lines.append("")
    return lines

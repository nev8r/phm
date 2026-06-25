"""
Feature analysis plots.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class PlotBuilder:
    def build(self, output_dir: Path, outputs: Dict) -> List[str]:
        payload = outputs.get("plot_payload") or {}
        if not payload.get("enabled", False):
            return []
        figures_dir = output_dir / "figures"
        curves_dir = figures_dir / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        top_k = int(payload.get("top_k", 10))
        max_bearings = int(payload.get("max_bearings", 5))
        features = payload.get("features")
        labels = payload.get("labels")
        fpt = payload.get("fpt")
        ranking = outputs.get("feature_ranking", pd.DataFrame())
        written: List[str] = []
        plot_frame = _merge_plot_frame(features, labels)
        selected_features = _selected_features(ranking, top_k)
        for feature in selected_features:
            written.extend(_plot_feature_curves(plot_frame, fpt, feature, curves_dir, max_bearings))
        written.append(_plot_top_bar(ranking, "rul_score", "Top RUL Features", figures_dir / "rul_top_features.png", top_k))
        written.append(_plot_degradation_heatmap(outputs, figures_dir / "degradation_score_heatmap.png", top_k))
        written.append(_plot_health_boxplots(plot_frame, ranking, figures_dir / "health_state_boxplots.png", top_k))
        written.append(_plot_early_fault_effects(plot_frame, ranking, figures_dir / "early_fault_effects.png", top_k))
        matrix_path = _plot_recommendation_matrix(ranking, figures_dir / "feature_recommendation_matrix.png", top_k)
        written.append(matrix_path)
        score_heatmap_path = figures_dir / "feature_score_heatmap.png"
        if matrix_path:
            _copy_png(figures_dir / "feature_recommendation_matrix.png", score_heatmap_path)
            written.append(str(score_heatmap_path))
        return [path for path in written if path]


def _merge_plot_frame(features: Optional[pd.DataFrame], labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    if features is None:
        return pd.DataFrame()
    data = features.copy()
    if labels is None or labels.empty:
        return data
    label_columns = [
        column for column in [
            "sample_uid",
            "piecewise_rul_norm",
            "linear_rul_norm",
            "health_state_id",
            "health_state_name",
            "early_fault",
            "fault_type_stage_id",
            "fault_type_stage_name",
        ]
        if column in labels.columns
    ]
    if label_columns == ["sample_uid"]:
        return data
    return data.merge(labels[label_columns], on="sample_uid", how="left")


def _selected_features(ranking: pd.DataFrame, top_k: int) -> List[str]:
    if ranking is None or ranking.empty:
        return []
    features = ranking.sort_values(["rank_overall", "feature"])["feature"].head(top_k).tolist()
    return [str(feature) for feature in features]


def _plot_feature_curves(data: pd.DataFrame, fpt: Optional[Dict], feature: str, curves_dir: Path, max_bearings: int) -> List[str]:
    if data.empty or feature not in data.columns:
        return []
    written: List[str] = []
    groups = list(data.groupby(["dataset", "bearing_id"], sort=False))[:max_bearings]
    aggregate_path = curves_dir / f"{_safe_name(feature)}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for (_, bearing_id), group in groups:
        group = group.sort_values("timestep")
        ax.plot(group["timestep"], pd.to_numeric(group[feature], errors="coerce"), marker="o", linewidth=1.5, label=str(bearing_id))
        plotted = True
    if plotted:
        ax.set_title(feature)
        ax.set_xlabel("timestep")
        ax.set_ylabel("feature value")
        ax.legend(loc="best", fontsize=8)
        _save(fig, aggregate_path)
        written.append(str(aggregate_path))
    else:
        written.append(_placeholder(aggregate_path, f"No curve data for {feature}"))
    fpt_lookup = _fpt_lookup(fpt)
    for (_, bearing_id), group in groups:
        group = group.sort_values("timestep")
        path = curves_dir / f"{_safe_name(str(bearing_id))}__{_safe_name(feature)}.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(group["timestep"], pd.to_numeric(group[feature], errors="coerce"), marker="o", linewidth=1.5)
        fpt_result = fpt_lookup.get((str(group["dataset"].iloc[0]), str(bearing_id)))
        if fpt_result is not None:
            ax.axvline(float(fpt_result["fpt_timestep"]), color="tab:red", linestyle="--", linewidth=1.2, label="FPT")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{bearing_id} - {feature}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("feature value")
        _save(fig, path)
        written.append(str(path))
    return written


def _plot_top_bar(ranking: pd.DataFrame, score_column: str, title: str, path: Path, top_k: int) -> str:
    if ranking is None or ranking.empty or score_column not in ranking.columns:
        return _placeholder(path, f"No data for {title}")
    data = ranking.sort_values(score_column, ascending=False).head(top_k)
    if data.empty:
        return _placeholder(path, f"No data for {title}")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(data))))
    ax.barh(data["feature"], data[score_column], color="#31688e")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(score_column)
    _save(fig, path)
    return str(path)


def _plot_degradation_heatmap(outputs: Dict, path: Path, top_k: int) -> str:
    ranking = outputs.get("feature_ranking", pd.DataFrame())
    rul = outputs.get("rul_correlation", pd.DataFrame())
    degradation = outputs.get("degradation_scores", pd.DataFrame())
    if ranking.empty:
        return _placeholder(path, "No degradation score data")
    rul_part = rul[["feature", "abs_spearman"]] if "abs_spearman" in rul.columns else pd.DataFrame(columns=["feature", "abs_spearman"])
    degradation_columns = [column for column in ["feature", "monotonicity", "robustness", "trendability"] if column in degradation.columns]
    degradation_part = degradation[degradation_columns] if "feature" in degradation_columns else pd.DataFrame(columns=["feature", "monotonicity", "robustness", "trendability"])
    data = ranking[["feature", "rul_score"]].merge(
        rul_part,
        on="feature",
        how="left",
    ).merge(
        degradation_part,
        on="feature",
        how="left",
    )
    data = data.sort_values("rul_score", ascending=False).head(top_k).set_index("feature").fillna(0.0)
    if data.empty:
        return _placeholder(path, "No degradation score data")
    return _heatmap(data, "Degradation Score Heatmap", path)


def _plot_health_boxplots(data: pd.DataFrame, ranking: pd.DataFrame, path: Path, top_k: int) -> str:
    if data.empty or "health_state_id" not in data.columns:
        return _placeholder(path, "No health-state labels")
    top_features = ranking.sort_values("health_score", ascending=False)["feature"].head(top_k).tolist()
    top_features = [feature for feature in top_features if feature in data.columns]
    if not top_features:
        return _placeholder(path, "No health-state feature data")
    melted = data.melt(
        id_vars=["health_state_name" if "health_state_name" in data.columns else "health_state_id"],
        value_vars=top_features,
        var_name="feature",
        value_name="value",
    )
    state_column = "health_state_name" if "health_state_name" in data.columns else "health_state_id"
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(top_features)), 5))
    sns.boxplot(data=melted, x="feature", y="value", hue=state_column, ax=ax)
    ax.set_title("Health State Boxplots")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)
    return str(path)


def _plot_early_fault_effects(data: pd.DataFrame, ranking: pd.DataFrame, path: Path, top_k: int) -> str:
    if data.empty or "early_fault" not in data.columns:
        return _placeholder(path, "No early-fault labels")
    top_features = ranking.sort_values("early_fault_score", ascending=False)["feature"].head(top_k).tolist()
    top_features = [feature for feature in top_features if feature in data.columns]
    if not top_features:
        return _placeholder(path, "No early-fault feature data")
    melted = data.melt(id_vars=["early_fault"], value_vars=top_features, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(top_features)), 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="early_fault", ax=ax)
    ax.set_title("Early Fault Effects")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, path)
    return str(path)


def _plot_recommendation_matrix(ranking: pd.DataFrame, path: Path, top_k: int) -> str:
    if ranking is None or ranking.empty:
        return _placeholder(path, "No recommendation data")
    columns = ["rul_score", "health_score", "early_fault_score", "fault_type_score"]
    data = ranking.sort_values("rank_overall").head(top_k).set_index("feature")[columns].fillna(0.0)
    if data.empty:
        return _placeholder(path, "No recommendation data")
    data = data.rename(columns={
        "rul_score": "RUL",
        "health_score": "HealthState",
        "early_fault_score": "EarlyFault",
        "fault_type_score": "FaultType",
    })
    return _heatmap(data, "Feature Recommendation Matrix", path)


def _heatmap(data: pd.DataFrame, title: str, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(data))))
    sns.heatmap(data, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    _save(fig, path)
    return str(path)


def _fpt_lookup(fpt: Optional[Dict]) -> Dict:
    if not fpt:
        return {}
    return {
        (str(result["dataset"]), str(result["bearing_id"])): result
        for result in fpt.get("results", [])
    }


def _placeholder(path: Path, message: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    _save(fig, path)
    return str(path)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _copy_png(source: Path, target: Path) -> None:
    target.write_bytes(source.read_bytes())

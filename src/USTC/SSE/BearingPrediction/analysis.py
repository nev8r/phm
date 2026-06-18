"""
Analysis utilities for bearing PHM experiments

this file is for defining dataset cards, label formulas, feature audits, and diagrams

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .data.paper.BearingPaperDataset import (
    DEFAULT_FREQUENCY_BANDS,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_TIME_FEATURES,
    PHM2012_LEARNING_BEARINGS,
    PHM2012_SAMPLING_RATE,
    XJTU_FAULT_LABELS,
    XJTU_SAMPLING_RATE,
    estimate_three_sigma_fot_index,
)


@dataclass(frozen=True)
class DatasetCard:
    """Compact dataset facts used by CLI reports and defense materials."""

    name: str
    sampling_rate_hz: int
    tasks: list[str]
    operating_conditions: list[str]
    bearing_count: int
    file_grain: str
    split_strategy: str
    label_source: str
    notes: list[str]


def build_dataset_cards() -> dict[str, dict[str, Any]]:
    """Return curated dataset cards for the two supported bearing datasets."""

    phm_card = DatasetCard(
        name="PHM2012 IEEE PHM Challenge bearing dataset",
        sampling_rate_hz=PHM2012_SAMPLING_RATE,
        tasks=["RUL"],
        operating_conditions=[
            "Condition 1: 1800 rpm / 4000 N",
            "Condition 2: 1650 rpm / 4200 N",
            "Condition 3: 1500 rpm / 5000 N",
        ],
        bearing_count=len(PHM2012_LEARNING_BEARINGS),
        file_grain="one vibration snapshot CSV per acquisition; horizontal and vertical channels",
        split_strategy="learning bearings for training/validation, full-test bearings for held-out evaluation",
        label_source="normalized run-to-failure progress, with optional rectified RUL plateau",
        notes=[
            "Each snapshot is converted to domain features before sequence modeling.",
            "The main reproduction target is continuous remaining useful life regression.",
        ],
    )
    xjtu_card = DatasetCard(
        name="XJTU-SY bearing accelerated life dataset",
        sampling_rate_hz=XJTU_SAMPLING_RATE,
        tasks=["Fault"],
        operating_conditions=[
            "Condition 1: 35 Hz / 12 kN",
            "Condition 2: 37.5 Hz / 11 kN",
            "Condition 3: 40 Hz / 10 kN",
        ],
        bearing_count=len(XJTU_FAULT_LABELS),
        file_grain="one vibration snapshot CSV per acquisition; horizontal and vertical channels",
        split_strategy="condition-aware bearing split with fixed random seed for train/validation/test",
        label_source="first fault occurrence time from horizontal RMS 3-sigma threshold",
        notes=[
            "The main reproduction target is binary healthy/faulty diagnosis.",
            "The threshold label is auditable and keeps the baseline comparison deterministic.",
        ],
    )
    return {"PHM2012": asdict(phm_card), "XJTU-SY": asdict(xjtu_card)}


def compute_rul_labels(
    length: int,
    mode: str = "linear",
    fpt_index: int | None = None,
) -> np.ndarray:
    """Compute normalized RUL labels.

    Linear mode uses y_t = 1 - t / (T - 1). Rectified mode keeps a healthy
    plateau until FPT and then decreases linearly to zero.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return np.array([1.0], dtype=np.float32)

    normalized = 1.0 - np.arange(length, dtype=np.float32) / float(length - 1)
    if mode == "linear":
        return normalized
    if mode != "rectified":
        raise ValueError(f"unsupported RUL label mode: {mode}")

    fpt = 0 if fpt_index is None else int(fpt_index)
    fpt = max(0, min(fpt, length - 1))
    labels = np.ones(length, dtype=np.float32)
    tail_length = length - fpt
    if tail_length <= 1:
        labels[-1] = 0.0
    else:
        labels[fpt:] = np.linspace(1.0, 0.0, tail_length, dtype=np.float32)
    return labels


def compute_fault_label_series(
    rms_values: np.ndarray | list[float],
    *,
    ratio: float = 3.0,
    healthy_ratio: float = 0.3,
    min_bound: float = 0.1,
    max_rms: float = 2.0,
    max_consecution: int = 5,
    consecution_ratio: float = 0.3,
) -> dict[str, Any]:
    """Compute binary labels from the 3-sigma first occurrence threshold."""

    values = np.asarray(rms_values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("rms_values must be a one-dimensional sequence")
    if values.size == 0:
        raise ValueError("rms_values must not be empty")

    healthy_count = max(1, int(values.size * healthy_ratio))
    healthy_values = values[:healthy_count]
    threshold = float(healthy_values.mean() + ratio * healthy_values.std() + min_bound)
    fot_index = estimate_three_sigma_fot_index(
        values,
        ratio=ratio,
        max_consecution=max_consecution,
        consecution_ratio=consecution_ratio,
        healthy_ratio=healthy_ratio,
        min_bound=min_bound,
        max_rms=max_rms,
    )
    labels = np.zeros(values.size, dtype=np.int64)
    if fot_index < values.size:
        labels[fot_index:] = 1
    return {
        "labels": labels,
        "fot_index": int(fot_index),
        "threshold": threshold,
        "formula": "FOT = first consecutive t where RMS_t > mean(RMS_healthy) + 3 sigma + 0.1 or RMS_t > 2.0",
        "healthy_count": healthy_count,
    }


def task_relationship_summary() -> dict[str, str]:
    """Return the shared mathematical view of RUL and fault diagnosis."""

    return {
        "shared_pipeline": "x(t,c) -> phi(x) -> X_i -> f_theta(X_i)",
        "rul_label": "y_t = 1 - t / (T - 1); rectified mode keeps y_t = 1 before FPT",
        "fault_label": "y_t = 0 before FOT, y_t = 1 after FOT",
        "relationship": "Fault diagnosis discretizes degradation state; RUL regression estimates continuous life progress.",
    }


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _monotonicity(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    diffs = np.diff(values)
    positive = np.sum(diffs > 0)
    negative = np.sum(diffs < 0)
    return float(abs(positive - negative) / max(1, diffs.size))


def compute_feature_analysis(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    task: str,
    top_k: int = 12,
) -> dict[str, Any]:
    """Summarize domain feature relevance and selection evidence."""

    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if matrix.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same sample count")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")

    start = perf_counter()
    correlations = [
        {
            "feature": str(name),
            "name": str(name),
            "index": index,
            "abs_pearson": _safe_abs_pearson(matrix[:, index], y),
            "monotonicity": _monotonicity(matrix[:, index]),
        }
        for index, name in enumerate(feature_names)
    ]
    correlations.sort(key=lambda item: item["abs_pearson"], reverse=True)

    centered = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    scaled = centered / (np.nanstd(centered, axis=0, keepdims=True) + 1e-12)
    corr_matrix = np.corrcoef(scaled, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = perf_counter() - start
    return {
        "task": task,
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "feature_strategy": "domain-first features with tsfresh audit and optional selection evidence",
        "top_correlated_features": correlations[:top_k],
        "top_correlations": correlations[:top_k],
        "correlation_heatmap": corr_matrix.tolist(),
        "why_not_default_tsfresh": [
            "tsfresh can generate hundreds of features per channel and raises full-data runtime and memory cost.",
            "The bearing task already has interpretable vibration features with direct physical meaning.",
            "We use tsfresh as an audit/selection reference instead of the default production feature set.",
        ],
        "elapsed_seconds": elapsed,
    }


def compute_tsfresh_audit(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    ids: np.ndarray | list[str] | None = None,
    times: np.ndarray | list[int] | None = None,
    mode: str = "minimal",
    max_domain_features: int = 12,
    top_k: int = 12,
) -> dict[str, Any]:
    """Run a bounded tsfresh audit over feature trajectories.

    The production feature set remains domain-first. This audit treats each
    selected domain feature trajectory as a time series kind and uses tsfresh to
    add independent selection evidence.
    """

    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="Deprecated call to `pkg_resources.declare_namespace.*",
            category=DeprecationWarning,
        )
        from tsfresh import extract_features
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    if matrix.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same sample count")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")
    if mode not in {"minimal", "efficient"}:
        raise ValueError(f"unsupported tsfresh audit mode: {mode}")

    ids_array = np.asarray(ids if ids is not None else np.repeat("series_0", matrix.shape[0])).astype(str)
    if ids_array.shape[0] != matrix.shape[0]:
        raise ValueError("ids length must match sample count")
    if times is None:
        time_values = pd.Series(ids_array).groupby(ids_array).cumcount().to_numpy()
    else:
        time_values = np.asarray(times)
        if time_values.shape[0] != matrix.shape[0]:
            raise ValueError("times length must match sample count")

    base_corr = [
        (index, _safe_abs_pearson(matrix[:, index], y))
        for index in range(matrix.shape[1])
    ]
    base_corr.sort(key=lambda item: item[1], reverse=True)
    selected_indices = [index for index, _ in base_corr[: max(1, min(max_domain_features, matrix.shape[1]))]]

    frames = []
    for index in selected_indices:
        frames.append(
            pd.DataFrame(
                {
                    "series_id": ids_array,
                    "time": time_values,
                    "kind": str(feature_names[index]),
                    "value": matrix[:, index],
                }
            )
        )
    long_frame = pd.concat(frames, ignore_index=True)
    fc_parameters = MinimalFCParameters() if mode == "minimal" else EfficientFCParameters()

    start = perf_counter()
    extracted = extract_features(
        long_frame,
        column_id="series_id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters,
        disable_progressbar=True,
        n_jobs=1,
    )
    elapsed = perf_counter() - start
    extracted = extracted.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    target_by_id = pd.DataFrame({"series_id": ids_array, "target": y}).groupby("series_id")["target"].mean()
    target_by_id = target_by_id.reindex(extracted.index).fillna(float(np.mean(y))).to_numpy()

    tsfresh_rank = []
    for index, column in enumerate(extracted.columns):
        values = extracted.iloc[:, index].to_numpy(dtype=np.float64)
        tsfresh_rank.append(
            {
                "feature": str(column),
                "abs_pearson": _safe_abs_pearson(values, target_by_id),
            }
        )
    tsfresh_rank.sort(key=lambda item: item["abs_pearson"], reverse=True)

    selected_names = [str(feature_names[index]) for index in selected_indices]
    top_names = [item["feature"] for item in tsfresh_rank[:top_k]]
    overlap = [
        name
        for name in selected_names
        if any(top_name.startswith(f"{name}__") for top_name in top_names)
    ]
    return {
        "mode": mode,
        "scope": "tsfresh audit over selected domain feature trajectories",
        "series_count": int(extracted.shape[0]),
        "selected_domain_feature_count": len(selected_names),
        "selected_domain_features": selected_names,
        "extracted_feature_count": int(extracted.shape[1]),
        "top_correlated_tsfresh_features": tsfresh_rank[:top_k],
        "domain_overlap": overlap,
        "input_rows": int(long_frame.shape[0]),
        "estimated_input_memory_mb": float(long_frame.memory_usage(deep=True).sum() / (1024 * 1024)),
        "elapsed_seconds": elapsed,
    }


def build_sample_feature_table(task: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a deterministic tiny feature table for CLI smoke tests."""

    rng = np.random.default_rng(2026 if task == "rul" else 2027)
    sample_count = 48 if task == "rul" else 54
    progress = np.linspace(0.0, 1.0, sample_count)
    rms = 0.4 + 0.8 * progress + rng.normal(0.0, 0.025, sample_count)
    kurtosis = 2.8 + 2.2 * np.maximum(progress - 0.35, 0.0) + rng.normal(0.0, 0.08, sample_count)
    peak_frequency = 1200.0 + 320.0 * np.sin(progress * math.pi) + rng.normal(0.0, 12.0, sample_count)
    band_energy = 0.2 + 1.5 * progress**2 + rng.normal(0.0, 0.04, sample_count)
    entropy = 0.7 - 0.2 * progress + rng.normal(0.0, 0.015, sample_count)
    features = np.column_stack([rms, kurtosis, peak_frequency, band_energy, entropy])
    if task == "rul":
        target = 1.0 - progress
    else:
        target = (progress >= 0.62).astype(np.float32)
    return features.astype(np.float32), target.astype(np.float32), [
        "rms",
        "kurtosis",
        "peak_frequency",
        "band_energy_3_6khz",
        "spectral_entropy",
    ]


def build_domain_feature_names(
    *,
    fft_bins: int,
    include_handcrafted: bool,
    channel_count: int,
) -> list[str]:
    """Reconstruct feature names from the loader's concatenation order."""

    names: list[str] = []
    for channel in range(channel_count):
        prefix = f"ch{channel}"
        names.extend([f"{prefix}_fft_bin_{index + 1:03d}" for index in range(fft_bins)])
        if include_handcrafted:
            names.extend([f"{prefix}_{name}" for name in DEFAULT_TIME_FEATURES])
            names.extend([f"{prefix}_{name}" for name in DEFAULT_SPECTRAL_FEATURES])
            names.extend(
                [
                    f"{prefix}_band_energy_{int(low)}_{int(high)}hz"
                    for low, high in DEFAULT_FREQUENCY_BANDS
                ]
            )
    return names


def render_feature_figures(
    output_dir: Path | str,
    analysis: dict[str, Any],
    feature_names: list[str] | tuple[str, ...],
    *,
    prefix: str,
) -> dict[str, Path]:
    """Render feature correlation and ranking figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranking = analysis.get("top_correlated_features", analysis["top_correlations"])
    top_items = ranking[: min(10, len(ranking))]
    ranking_path = out / f"{prefix}_feature_rank.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names = [item["feature"] for item in reversed(top_items)]
    values = [item["abs_pearson"] for item in reversed(top_items)]
    ax.barh(names, values, color="#1f77b4")
    ax.set_xlabel("|Pearson correlation|")
    ax.set_title(f"{prefix.upper()} domain feature relevance")
    ax.set_xlim(0.0, max(1.0, max(values, default=0.0) * 1.05))
    fig.tight_layout()
    fig.savefig(ranking_path, dpi=180)
    plt.close(fig)

    heatmap_path = out / f"{prefix}_feature_heatmap.png"
    full_matrix = np.asarray(analysis["correlation_heatmap"], dtype=np.float64)
    selected_indices = [int(item.get("index", 0)) for item in top_items[:20]]
    if not selected_indices:
        selected_indices = list(range(min(20, len(feature_names))))
    matrix = full_matrix[np.ix_(selected_indices, selected_indices)]
    selected_names = [str(feature_names[index]) for index in selected_indices]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(selected_names)))
    ax.set_yticks(range(len(selected_names)))
    ax.set_xticklabels(selected_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(selected_names, fontsize=7)
    ax.set_title(f"{prefix.upper()} feature correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    return {"ranking": ranking_path, "heatmap": heatmap_path}


def render_tsfresh_audit_figures(
    output_dir: Path | str,
    audit: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Path]:
    """Render mode-specific tsfresh audit figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mode = str(audit.get("mode", "minimal"))
    ranking = list(audit.get("top_correlated_tsfresh_features", []))
    top_items = ranking[: min(12, len(ranking))]

    rank_path = out / f"{prefix}_tsfresh_{mode}_rank.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    names = [str(item.get("feature", "")) for item in reversed(top_items)]
    values = [float(item.get("abs_pearson", 0.0)) for item in reversed(top_items)]
    if names:
        ax.barh(names, values, color="#0f766e")
    else:
        ax.text(0.5, 0.5, "No tsfresh features", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("|Pearson correlation with target|")
    ax.set_title(f"{prefix.upper()} tsfresh {mode} feature relevance")
    ax.set_xlim(0.0, max(1.0, max(values, default=0.0) * 1.05))
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(rank_path, dpi=180)
    plt.close(fig)

    profile_path = out / f"{prefix}_tsfresh_{mode}_profile.png"
    profile = {
        "selected domain": float(audit.get("selected_domain_feature_count", 0)),
        "extracted tsfresh": float(audit.get("extracted_feature_count", 0)),
        "series": float(audit.get("series_count", 0)),
        "memory MB": float(audit.get("estimated_input_memory_mb", 0.0)),
        "seconds": float(audit.get("elapsed_seconds", 0.0)),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = list(profile)
    values = list(profile.values())
    ax.bar(labels, values, color=["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#dc2626"])
    ax.set_title(f"{prefix.upper()} tsfresh {mode} audit profile")
    ax.set_ylabel("count / MB / seconds")
    ax.tick_params(axis="x", labelrotation=20)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.2g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)
    return {"rank": rank_path, "profile": profile_path}


def render_model_architecture_diagrams(output_dir: Path | str) -> dict[str, Path]:
    """Generate stable PNG architecture diagrams without Graphviz."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = {
        "rul": {
            "filename": "rul_model_architecture.png",
            "title": "CBAM-CNN-LSTM RUL regression",
            "boxes": [
                "Input\n[B, 32, F]",
                "1D CNN\nlocal spectral patterns",
                "CBAM\nchannel + temporal attention",
                "LSTM\ndegradation sequence",
                "Regression head\nnormalized RUL",
                "Loss/metrics\nMSE, RMSE, MAE, R2",
            ],
        },
        "fault": {
            "filename": "fault_model_architecture.png",
            "title": "ResCNN-LSTM fault diagnosis",
            "boxes": [
                "Input\n[B, 8, F]",
                "Residual CNN\nstable local features",
                "LSTM\nsnapshot context",
                "Classifier head\nhealthy/faulty",
                "Loss/metrics\nCrossEntropy, F1, confusion matrix",
            ],
        },
    }
    paths: dict[str, Path] = {}
    for key, spec in specs.items():
        box_count = len(spec["boxes"])
        fig_width = max(10.0, box_count * 2.15)
        fig, ax = plt.subplots(figsize=(fig_width, 3.2))
        ax.set_axis_off()
        ax.set_xlim(0.0, box_count)
        ax.set_ylim(0.0, 1.0)
        ax.text(0.02, 0.94, spec["title"], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        for index, label in enumerate(spec["boxes"]):
            x = index + 0.08
            patch = FancyBboxPatch(
                (x, 0.30),
                0.84,
                0.34,
                boxstyle="round,pad=0.035,rounding_size=0.025",
                linewidth=1.4,
                edgecolor="#1f4e79",
                facecolor="#eef5ff",
            )
            ax.add_patch(patch)
            ax.text(x + 0.42, 0.47, label, ha="center", va="center", fontsize=9.5)
            if index < box_count - 1:
                arrow = FancyArrowPatch(
                    (x + 0.86, 0.47),
                    (index + 1.08, 0.47),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=1.2,
                    color="#1f4e79",
                )
                ax.add_patch(arrow)
        path = out / spec["filename"]
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[key] = path
    return paths


def json_safe(data: Any) -> Any:
    """Convert numpy/path objects to JSON serializable values."""

    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(value) for value in data]
    return data


def write_json(path: Path | str, data: Any) -> None:
    """Write deterministic UTF-8 JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")

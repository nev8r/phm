"""
Prediction sanity audit for completed PHM baseline runs.

Purpose: provide reproducible demo or diagnostic workflow for 轴承寿命预测与故障诊断系统
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EXPERIMENTS = [
    "xjtu_main_rul_mlp_full_manual_basic_no_reference",
    "phm_official_rul_mlp_compact_non_label_source",
    "phm_official_rul_mlp_tuned_compact_non_label_source",
    "y01_xjtu_rul_xgboost_full_manual_basic_no_reference",
    "y02_xjtu_rul_random_forest_full_manual_basic_no_reference",
    "y07_phm_rul_xgboost_compact_non_label_source",
    "y08_phm_rul_random_forest_compact_non_label_source",
    "xjtu_main_health_mlp_compact_non_label_source",
    "xjtu_main_early_mlp_compact_non_label_source",
    "y03_xjtu_health_xgboost_compact_non_label_source",
    "y05_xjtu_early_xgboost_compact_non_label_source",
    "y09_phm_health_xgboost_compact_non_label_source",
    "y11_phm_early_xgboost_compact_non_label_source",
]

REGRESSION = "regression"
CLASSIFICATION_TYPES = {"binary_classification", "multiclass_classification"}


@dataclass
class ExperimentContext:
    experiment_id: str
    source_kind: str
    raw_dir: Path
    report_dir: Path | None
    dataset: str
    task: str
    task_type: str
    model_family: str
    target_columns: List[str]
    raw_run_id: str


def normalize_prediction_frame(frame: pd.DataFrame, target_columns: Sequence[str]) -> pd.DataFrame:
    """Normalize MLP and sklearn prediction tables to y_true/y_pred columns."""
    normalized = frame.copy()
    if "target_sample_uid" not in normalized.columns and "sample_uid" in normalized.columns:
        normalized["target_sample_uid"] = normalized["sample_uid"]
    if "sample_uid" not in normalized.columns and "target_sample_uid" in normalized.columns:
        normalized["sample_uid"] = normalized["target_sample_uid"]

    if "y_true" not in normalized.columns:
        true_column = _prediction_column(normalized, "y_true", target_columns)
        normalized["y_true"] = normalized[true_column]
    if "y_pred" not in normalized.columns:
        pred_column = _prediction_column(normalized, "y_pred", target_columns)
        normalized["y_pred"] = normalized[pred_column]

    for column in ["sample_uid", "target_sample_uid"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype(str)
    return normalized


def alignment_check_row(
    *,
    experiment_id: str,
    dataset: str,
    task: str,
    task_type: str,
    model_family: str,
    split: str,
    predictions: pd.DataFrame,
    labels: pd.DataFrame | None,
    target_column: str,
    manifest_available: bool,
) -> Dict[str, Any]:
    key = "target_sample_uid" if "target_sample_uid" in predictions.columns else "sample_uid"
    duplicate_count = int(predictions[key].duplicated().sum()) if key in predictions.columns else len(predictions)
    labels_available = labels is not None and not labels.empty and "sample_uid" in labels.columns and target_column in labels.columns
    missing_labels = len(predictions)
    mismatches = len(predictions)
    if labels_available:
        label_values = labels[["sample_uid", target_column]].copy()
        label_values["sample_uid"] = label_values["sample_uid"].astype(str)
        merged = predictions[[key, "y_true"]].merge(
            label_values,
            left_on=key,
            right_on="sample_uid",
            how="left",
            suffixes=("", "__label"),
        )
        missing_labels = int(merged[target_column].isna().sum())
        comparable = merged[merged[target_column].notna()]
        mismatches = int((np.abs(comparable["y_true"].astype(float) - comparable[target_column].astype(float)) > 1.0e-6).sum())

    monotonic = _target_timestep_monotonic(predictions)
    alignment_ok = duplicate_count == 0 and missing_labels == 0 and mismatches == 0 and monotonic
    return {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "task": task,
        "task_type": task_type,
        "model_family": model_family,
        "split": split,
        "num_prediction_rows": int(len(predictions)),
        "num_missing_labels": int(missing_labels),
        "num_duplicate_sample_uid": int(duplicate_count),
        "num_mismatched_targets": int(mismatches),
        "target_timestep_monotonic_by_bearing": "yes" if monotonic else "no",
        "manifest_available": "yes" if manifest_available else "no",
        "labels_available": "yes" if labels_available else "no",
        "alignment_ok": "yes" if alignment_ok else "no",
    }


def rul_range_check_row(
    *,
    experiment_id: str,
    dataset: str,
    task: str,
    model_family: str,
    split: str,
    predictions: pd.DataFrame,
) -> Dict[str, Any]:
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    clipped = np.clip(y_pred, 0.0, 1.0)
    raw_rmse = _rmse(y_true, y_pred)
    clipped_rmse = _rmse(y_true, clipped)
    raw_mae = _mae(y_true, y_pred)
    clipped_mae = _mae(y_true, clipped)
    out_of_range = (y_pred < 0.0) | (y_pred > 1.0)
    return {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "task": task,
        "model_family": model_family,
        "split": split,
        "n": int(len(predictions)),
        "y_true_min": float(np.min(y_true)) if len(y_true) else math.nan,
        "y_true_max": float(np.max(y_true)) if len(y_true) else math.nan,
        "y_pred_min": float(np.min(y_pred)) if len(y_pred) else math.nan,
        "y_pred_max": float(np.max(y_pred)) if len(y_pred) else math.nan,
        "num_pred_lt_0": int((y_pred < 0.0).sum()),
        "num_pred_gt_1": int((y_pred > 1.0).sum()),
        "clip_rate": float(out_of_range.mean()) if len(out_of_range) else math.nan,
        "raw_RMSE": raw_rmse,
        "clipped_RMSE": clipped_rmse,
        "raw_MAE": raw_mae,
        "clipped_MAE": clipped_mae,
        "rmse_improvement_after_clip": float(raw_rmse - clipped_rmse),
        "clip_improves_rmse": "yes" if clipped_rmse + 1.0e-12 < raw_rmse else "no",
    }


def naive_baseline_rows(
    *,
    experiment_id: str,
    dataset: str,
    task: str,
    task_type: str,
    model_family: str,
    train_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if task_type == REGRESSION:
        y_train = train_predictions["y_true"].to_numpy(dtype=float)
        y_true = test_predictions["y_true"].to_numpy(dtype=float)
        y_pred = test_predictions["y_pred"].to_numpy(dtype=float)
        model_metric = _rmse(y_true, y_pred)
        strategies = {
            "train_mean": np.full(len(y_true), float(np.mean(y_train))),
            "train_median": np.full(len(y_true), float(np.median(y_train))),
            "constant_zero": np.zeros(len(y_true), dtype=float),
            "constant_one": np.ones(len(y_true), dtype=float),
        }
        return [
            {
                "experiment_id": experiment_id,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model_family": model_family,
                "naive_strategy": strategy,
                "primary_metric": "RMSE",
                "metric_direction": "lower_is_better",
                "model_test_metric": model_metric,
                "naive_test_metric": _rmse(y_true, naive_pred),
                "model_beats_naive": "yes" if model_metric < _rmse(y_true, naive_pred) else "no",
            }
            for strategy, naive_pred in strategies.items()
        ]

    if task_type in CLASSIFICATION_TYPES:
        y_train = train_predictions["y_true"].to_numpy(dtype=int)
        y_true = test_predictions["y_true"].to_numpy(dtype=int)
        y_pred = test_predictions["y_pred"].to_numpy(dtype=int)
        model_metric = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        majority = int(pd.Series(y_train).value_counts().sort_values(ascending=False).index[0])
        naive_pred = np.full(len(y_true), majority, dtype=int)
        naive_metric = float(f1_score(y_true, naive_pred, average="weighted", zero_division=0))
        return [{
            "experiment_id": experiment_id,
            "dataset": dataset,
            "task": task,
            "task_type": task_type,
            "model_family": model_family,
            "naive_strategy": f"majority_class_{majority}",
            "primary_metric": "WeightedF1",
            "metric_direction": "higher_is_better",
            "model_test_metric": model_metric,
            "naive_test_metric": naive_metric,
            "model_beats_naive": "yes" if model_metric > naive_metric else "no",
        }]

    raise ValueError(f"Unsupported task_type: {task_type}")


def per_bearing_metric_rows(
    *,
    experiment_id: str,
    dataset: str,
    task: str,
    task_type: str,
    model_family: str,
    split: str,
    predictions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if "bearing_id" not in predictions.columns:
        return rows
    for bearing_id, group in predictions.groupby("bearing_id", dropna=False):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        if task_type == REGRESSION:
            metrics = {"RMSE": _rmse(y_true.astype(float), y_pred.astype(float)), "MAE": _mae(y_true.astype(float), y_pred.astype(float))}
        elif task_type in CLASSIFICATION_TYPES:
            metrics = {
                "Accuracy": float(accuracy_score(y_true.astype(int), y_pred.astype(int))),
                "WeightedF1": float(f1_score(y_true.astype(int), y_pred.astype(int), average="weighted", zero_division=0)),
            }
        else:
            metrics = {}
        for metric, value in metrics.items():
            rows.append({
                "dataset": dataset,
                "task": task,
                "experiment_id": experiment_id,
                "model_family": model_family,
                "split": split,
                "bearing_id": str(bearing_id),
                "n": int(len(group)),
                "metric": metric,
                "value": float(value),
            })
    return rows


def classification_per_class_rows(
    *,
    experiment_id: str,
    dataset: str,
    task: str,
    model_family: str,
    split: str,
    predictions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows = []
    for index, class_id in enumerate(labels):
        rows.append({
            "experiment_id": experiment_id,
            "dataset": dataset,
            "task": task,
            "model_family": model_family,
            "split": split,
            "class_id": int(class_id),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        })
    return rows


def run_audit(
    *,
    baseline_root: Path,
    non_mlp_root: Path,
    artifact_root: Path,
    output: Path,
    experiments: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    if output.exists():
        shutil.rmtree(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    contexts = [
        build_experiment_context(
            experiment_id=experiment_id,
            baseline_root=baseline_root,
            non_mlp_root=non_mlp_root,
            artifact_root=artifact_root,
        )
        for experiment_id in experiments
    ]

    alignment_rows: List[Dict[str, Any]] = []
    range_rows: List[Dict[str, Any]] = []
    naive_rows: List[Dict[str, Any]] = []
    bearing_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for context in contexts:
        split_predictions = load_split_predictions(context)
        sample_uids = sorted({
            sample_uid
            for frame in split_predictions.values()
            for sample_uid in frame["target_sample_uid"].astype(str).tolist()
        })
        labels = load_labels(context.raw_dir, artifact_root, context.target_columns[0], sample_uids)
        manifest_available = (context.raw_dir / "task" / "task_manifest.parquet").exists()
        train_predictions = split_predictions.get("train")
        if train_predictions is None:
            train_predictions = train_predictions_from_manifest(context.raw_dir, labels, context.target_columns[0])

        for split, predictions in split_predictions.items():
            alignment_rows.append(alignment_check_row(
                experiment_id=context.experiment_id,
                dataset=context.dataset,
                task=context.task,
                task_type=context.task_type,
                model_family=context.model_family,
                split=split,
                predictions=predictions,
                labels=labels,
                target_column=context.target_columns[0],
                manifest_available=manifest_available,
            ))
            bearing_rows.extend(per_bearing_metric_rows(
                experiment_id=context.experiment_id,
                dataset=context.dataset,
                task=context.task,
                task_type=context.task_type,
                model_family=context.model_family,
                split=split,
                predictions=predictions,
            ))
            if context.task_type == REGRESSION:
                range_rows.append(rul_range_check_row(
                    experiment_id=context.experiment_id,
                    dataset=context.dataset,
                    task=context.task,
                    model_family=context.model_family,
                    split=split,
                    predictions=predictions,
                ))
                if split in {"val", "test"}:
                    plot_rul_true_pred_by_bearing(
                        output / "figures" / f"{context.experiment_id}_{split}_true_pred_by_bearing.png",
                        predictions,
                        title=f"{context.experiment_id} - {split} true/pred by bearing",
                    )
            elif context.task_type in CLASSIFICATION_TYPES:
                class_rows.extend(classification_per_class_rows(
                    experiment_id=context.experiment_id,
                    dataset=context.dataset,
                    task=context.task,
                    model_family=context.model_family,
                    split=split,
                    predictions=predictions,
                ))

        if train_predictions is not None and "test" in split_predictions:
            naive_rows.extend(naive_baseline_rows(
                experiment_id=context.experiment_id,
                dataset=context.dataset,
                task=context.task,
                task_type=context.task_type,
                model_family=context.model_family,
                train_predictions=train_predictions,
                test_predictions=split_predictions["test"],
            ))

    outputs = {
        "prediction_alignment_check": pd.DataFrame(alignment_rows),
        "rul_prediction_range_check": pd.DataFrame(range_rows),
        "naive_baseline_comparison": pd.DataFrame(naive_rows),
        "per_bearing_metrics": pd.DataFrame(bearing_rows),
        "classification_per_class_metrics": pd.DataFrame(class_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    write_readme(output)
    write_runs_and_manifest(output)
    write_summary_report(output, outputs, contexts)
    return outputs


def build_experiment_context(
    *,
    experiment_id: str,
    baseline_root: Path,
    non_mlp_root: Path,
    artifact_root: Path,
) -> ExperimentContext:
    baseline_report_dir = baseline_root / experiment_id
    non_mlp_report_dir = non_mlp_root / experiment_id
    if baseline_report_dir.exists():
        source_kind = "mlp"
        report_dir: Path | None = baseline_report_dir
        raw_dir = latest_run_dir(artifact_root / "baselines" / "runs", experiment_id)
    elif non_mlp_report_dir.exists():
        source_kind = "non_mlp"
        report_dir = non_mlp_report_dir
        raw_dir = latest_run_dir(artifact_root / "non_mlp_baselines" / "runs", experiment_id)
    else:
        raw_dir = latest_run_dir(artifact_root / "baselines" / "runs", experiment_id, required=False)
        source_kind = "mlp"
        report_dir = None
        if raw_dir is None:
            raw_dir = latest_run_dir(artifact_root / "non_mlp_baselines" / "runs", experiment_id)
            source_kind = "non_mlp"

    task_spec = load_json(first_existing([
        raw_dir / "task" / "task_spec.json",
        raw_dir / "task_spec.json",
        report_dir / "task_spec.json" if report_dir is not None else None,
    ]))
    metrics = load_json(first_existing([
        raw_dir / "metrics.json",
        report_dir / "metrics.json" if report_dir is not None else None,
        report_dir / "test_metrics.json" if report_dir is not None else None,
    ], required=False))
    dataset = str(metrics.get("dataset") or _dataset_from_predictions(raw_dir) or _dataset_from_experiment_id(experiment_id))
    model_family = str(metrics.get("model_family") or metrics.get("model") or ("MLP" if source_kind == "mlp" else "non-MLP"))
    return ExperimentContext(
        experiment_id=experiment_id,
        source_kind=source_kind,
        raw_dir=raw_dir,
        report_dir=report_dir,
        dataset=dataset,
        task=str(task_spec.get("name", metrics.get("task", ""))),
        task_type=str(task_spec["task_type"]),
        model_family=model_family,
        target_columns=list(task_spec.get("target_columns", [])) or read_lines(first_existing([
            raw_dir / "task" / "target_columns.txt",
            raw_dir / "target_columns.txt",
            report_dir / "target_columns.txt" if report_dir is not None else None,
        ], required=False)),
        raw_run_id=raw_dir.name,
    )


def latest_run_dir(root: Path, experiment_id: str, required: bool = True) -> Path | None:
    candidates = sorted(path for path in root.glob(f"*{experiment_id}*") if path.is_dir())
    if not candidates:
        if required:
            raise FileNotFoundError(f"No artifact run found for {experiment_id} under {root}")
        return None
    return candidates[-1]


def load_split_predictions(context: ExperimentContext) -> Dict[str, pd.DataFrame]:
    predictions: Dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        path = context.raw_dir / "predictions" / f"{split}_predictions.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        predictions[split] = normalize_prediction_frame(frame, context.target_columns)
    if not predictions:
        raise FileNotFoundError(f"No prediction parquet files found for {context.experiment_id}")
    return predictions


def load_labels(raw_dir: Path, artifact_root: Path, target_column: str, sample_uids: Sequence[str]) -> pd.DataFrame | None:
    direct = raw_dir / "labels" / "labels.parquet"
    if direct.exists():
        return pd.read_parquet(direct)
    wanted = set(str(uid) for uid in sample_uids)
    for labels_path in label_candidates(artifact_root):
        try:
            labels = pd.read_parquet(labels_path)
        except Exception:
            continue
        if "sample_uid" not in labels.columns or target_column not in labels.columns:
            continue
        available = set(labels["sample_uid"].astype(str).tolist())
        if wanted.issubset(available):
            return labels
    return None


def label_candidates(artifact_root: Path) -> List[Path]:
    roots = [
        artifact_root / "baselines" / "runs",
        artifact_root / "feature_analysis" / "runs",
        artifact_root / "baseline_preflight" / "runs",
    ]
    paths: List[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.glob("*/labels/labels.parquet"))
    return sorted(paths, reverse=True)


def train_predictions_from_manifest(raw_dir: Path, labels: pd.DataFrame | None, target_column: str) -> pd.DataFrame | None:
    manifest_path = raw_dir / "task" / "task_manifest.parquet"
    if labels is None or not manifest_path.exists() or target_column not in labels.columns:
        return None
    manifest = pd.read_parquet(manifest_path)
    train = manifest[manifest["split"] == "train"].copy()
    if train.empty:
        return None
    labels_by_uid = labels[["sample_uid", target_column]].copy()
    labels_by_uid["sample_uid"] = labels_by_uid["sample_uid"].astype(str)
    train["target_sample_uid"] = train["target_sample_uid"].astype(str)
    merged = train.merge(labels_by_uid, left_on="target_sample_uid", right_on="sample_uid", how="left", suffixes=("", "__label"))
    if merged[target_column].isna().any():
        return None
    frame = merged[[
        "example_uid",
        "split",
        "dataset",
        "bearing_id",
        "condition_id",
        "source_group",
        "target_sample_uid",
        "target_timestep",
    ]].copy()
    frame["sample_uid"] = frame["target_sample_uid"]
    frame["y_true"] = merged[target_column].astype(float)
    return frame


def plot_rul_true_pred_by_bearing(path: Path, predictions: pd.DataFrame, title: str) -> None:
    bearings = list(predictions.groupby("bearing_id", dropna=False).groups.keys())
    columns = min(3, max(1, len(bearings)))
    rows = int(math.ceil(len(bearings) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.2 * rows), dpi=160, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, bearing_id in zip(axes.flat, bearings):
        group = predictions[predictions["bearing_id"] == bearing_id].sort_values("target_timestep")
        ax.axis("on")
        ax.plot(group["target_timestep"], group["y_true"], color="#2B6CB0", linewidth=1.6, label="true")
        ax.plot(group["target_timestep"], group["y_pred"], color="#C53030", linewidth=1.3, alpha=0.9, label="pred")
        ax.set_title(str(bearing_id))
        ax.set_xlabel("target_timestep")
        ax.set_ylabel("piecewise_rul_norm")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary_report(output: Path, outputs: Dict[str, pd.DataFrame], contexts: Sequence[ExperimentContext]) -> None:
    alignment = outputs["prediction_alignment_check"]
    range_check = outputs["rul_prediction_range_check"]
    naive = outputs["naive_baseline_comparison"]
    per_bearing = outputs["per_bearing_metrics"]
    per_class = outputs["classification_per_class_metrics"]

    alignment_bad = alignment[alignment["alignment_ok"] != "yes"] if not alignment.empty else alignment
    out_of_range = range_check[range_check["clip_rate"] > 0.0] if not range_check.empty else range_check
    clipped_help = range_check[range_check["clip_improves_rmse"] == "yes"] if not range_check.empty else range_check
    naive_fail = naive[naive["model_beats_naive"] != "yes"] if not naive.empty else naive
    worst_rul = _worst_per_bearing(per_bearing, task_type="rul")
    worst_cls = _worst_per_bearing(per_bearing, task_type="classification")

    lines = [
        "# Prediction Sanity Audit",
        "",
        "## 1. Scope",
        "",
        "Step Y-D pauses GUI/demo work and audits completed prediction outputs. No new model training is run.",
        "",
        f"- Audited experiments: {len(contexts)}",
        f"- Alignment rows: {len(alignment)}",
        f"- RUL range rows: {len(range_check)}",
        f"- Naive comparison rows: {len(naive)}",
        f"- Per-bearing metric rows: {len(per_bearing)}",
        f"- Per-class metric rows: {len(per_class)}",
        "",
        "## 2. Direct Answers",
        "",
        f"1. sample_uid / target 对齐错误：{'发现，需要先处理' if not alignment_bad.empty else '未发现。所有可验证 prediction y_true 均和 labels 表一致。'}",
        f"2. RUL y_pred 越界：{'存在' if not out_of_range.empty else '未发现明显越界'}。",
        f"3. clipped RMSE 是否显著改善：{'部分 split 改善，说明输出范围约束会影响 RMSE 解读' if not clipped_help.empty else '未出现实质改善，主要偏差不是越界造成的'}。",
        f"4. 模型是否打败 naive baseline：{'存在未打败 naive 的实验/策略，需降级解读' if not naive_fail.empty else '所有审计实验在主 naive 策略上均有优势或至少不弱'}。",
        f"5. 严重偏离集中在哪些 bearing：见 per_bearing_metrics.csv；RUL 最差条目见下表。",
        "6. 初步归因：若 alignment 通过且 train 明显优于 val/test，优先解释为过拟合、bearing/condition 分布偏移和特征泛化不足；若 clip 改善有限，则不是单纯输出越界问题。",
        "7. 既有报告结论：baseline 排名可以保留为实验记录，但所有 RUL 效果描述必须降级为 early baseline，不应宣称预测质量已经稳定。",
        "8. GUI 状态：暂停最终展示。GUI 可以作为过程演示工具，但不应继续包装为模型效果证明，直到本审计通过验收并完成后续修正。",
        "",
        "## 3. Alignment Summary",
        "",
        _markdown_table(_compact_alignment(alignment)),
        "",
        "## 4. RUL Range Summary",
        "",
        _markdown_table(_compact_range(range_check)),
        "",
        "## 5. Naive Baseline Summary",
        "",
        _markdown_table(_compact_naive(naive)),
        "",
        "## 6. Worst Per-Bearing Rows",
        "",
        "### RUL",
        "",
        _markdown_table(worst_rul),
        "",
        "### Classification",
        "",
        _markdown_table(worst_cls),
        "",
        "## 7. Generated Figures",
        "",
        "- `figures/*_val_true_pred_by_bearing.png`",
        "- `figures/*_test_true_pred_by_bearing.png`",
        "",
        "## 8. Decision",
        "",
        "- [x] Step Y-D audit artifacts generated",
        "- [x] No new training was run",
        "- [x] GUI remains paused as a model-quality claim",
        "- [ ] Ready to resume final GUI/video narrative",
        "",
    ]
    (output / "00_prediction_sanity_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# Prediction Audit

This directory contains Step Y-D prediction sanity audit outputs.

The audit reads existing MLP and non-MLP prediction artifacts, checks label alignment, RUL prediction range, clipped metrics, naive baselines, per-bearing metrics, per-class classification metrics, and true-vs-pred RUL curves by bearing.

No model training is performed in this step.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_runs_and_manifest(output: Path) -> None:
    runs = """# Prediction Audit Runs

| Step | Type | Description | Status |
|---|---|---|---|
| Step Y-D | QA | Prediction sanity audit for completed MLP and non-MLP baselines | needs-review |
"""
    manifest = pd.DataFrame([{
        "step": "StepY-D",
        "type": "QA",
        "name": "prediction_sanity_audit",
        "source": "artifacts/baselines;artifacts/non_mlp_baselines;reports/baseline_results;reports/non_mlp_baseline_results",
        "artifact": "reports/prediction_audit",
        "status": "needs-review",
        "notes": "Audits existing predictions for alignment, RUL range, clipped RMSE, naive baselines, per-bearing metrics, per-class metrics, and true/pred curves; no new training.",
    }])
    (output / "RUNS.md").write_text(runs, encoding="utf-8")
    manifest.to_csv(output / "MANIFEST.csv", index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completed PHM prediction outputs.")
    parser.add_argument("--baseline-root", type=Path, default=Path("reports/baseline_results"))
    parser.add_argument("--non-mlp-root", type=Path, default=Path("reports/non_mlp_baseline_results"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/prediction_audit"))
    parser.add_argument("--experiments", type=str, default=",".join(DEFAULT_EXPERIMENTS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    outputs = run_audit(
        baseline_root=args.baseline_root,
        non_mlp_root=args.non_mlp_root,
        artifact_root=args.artifact_root,
        output=args.output,
        experiments=experiments,
    )
    print(f"Prediction audit completed: {args.output}")
    for name, frame in outputs.items():
        print(f"- {name}: {len(frame)} rows")


def _prediction_column(frame: pd.DataFrame, prefix: str, target_columns: Sequence[str]) -> str:
    for target in target_columns:
        candidate = f"{prefix}__{target}"
        if candidate in frame.columns:
            return candidate
    candidates = [column for column in frame.columns if column.startswith(f"{prefix}__")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not find {prefix} column in prediction frame")


def _target_timestep_monotonic(predictions: pd.DataFrame) -> bool:
    if "bearing_id" not in predictions.columns or "target_timestep" not in predictions.columns:
        return False
    for _, group in predictions.groupby("bearing_id", dropna=False):
        if not group["target_timestep"].is_monotonic_increasing:
            return False
    return True


def _rmse(y_true: Iterable, y_pred: Iterable) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((pred - true) ** 2))) if len(true) else math.nan


def _mae(y_true: Iterable, y_pred: Iterable) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(pred - true))) if len(true) else math.nan


def load_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(paths: Sequence[Path | None], required: bool = True) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    if required:
        raise FileNotFoundError("None of the candidate paths exists")
    return None


def read_lines(path: Path | None) -> List[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset_from_predictions(raw_dir: Path) -> str | None:
    for split in ["test", "val", "train"]:
        path = raw_dir / "predictions" / f"{split}_predictions.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=["dataset"])
            if not frame.empty:
                return str(frame["dataset"].iloc[0])
    return None


def _dataset_from_experiment_id(experiment_id: str) -> str:
    if experiment_id.startswith("xjtu") or "_xjtu_" in experiment_id:
        return "XJTU-SY"
    if experiment_id.startswith("phm") or "_phm_" in experiment_id:
        return "PHM2012"
    return ""


def _worst_per_bearing(per_bearing: pd.DataFrame, task_type: str) -> pd.DataFrame:
    if per_bearing.empty:
        return pd.DataFrame()
    if task_type == "rul":
        subset = per_bearing[(per_bearing["task"] == "rul_tabular") & (per_bearing["split"] == "test") & (per_bearing["metric"] == "RMSE")]
        return subset.sort_values("value", ascending=False).head(10)
    subset = per_bearing[(per_bearing["task"] != "rul_tabular") & (per_bearing["split"] == "test") & (per_bearing["metric"] == "WeightedF1")]
    return subset.sort_values("value", ascending=True).head(10)


def _compact_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "experiment_id",
        "split",
        "num_prediction_rows",
        "num_missing_labels",
        "num_duplicate_sample_uid",
        "num_mismatched_targets",
        "alignment_ok",
    ]
    return frame[columns].sort_values(["alignment_ok", "experiment_id", "split"]).head(20)


def _compact_range(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "experiment_id",
        "split",
        "y_pred_min",
        "y_pred_max",
        "clip_rate",
        "raw_RMSE",
        "clipped_RMSE",
        "clip_improves_rmse",
    ]
    return frame[columns].sort_values(["clip_rate", "experiment_id"], ascending=[False, True]).head(20)


def _compact_naive(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "experiment_id",
        "naive_strategy",
        "primary_metric",
        "model_test_metric",
        "naive_test_metric",
        "model_beats_naive",
    ]
    return frame[columns].sort_values(["model_beats_naive", "experiment_id", "naive_strategy"]).head(30)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = [str(column) for column in display.columns]
    rows = [[_markdown_cell(value) for value in row] for row in display.itertuples(index=False, name=None)]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()

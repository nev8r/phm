"""
Curate Step AB GRU sequence training outputs.

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
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SequenceRunPlan:
    experiment_id: str
    dataset: str
    task: str
    task_type: str
    target: str
    model: str
    feature_subset: str
    feature_count: int
    sequence_length: int
    label_source_included: str
    command: str


AB_RUNS: List[SequenceRunPlan] = [
    SequenceRunPlan(
        experiment_id="xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep",
        dataset="XJTU-SY",
        task="rul_linear_sequence",
        task_type="regression",
        target="linear_rul_norm",
        model="gru",
        feature_subset="full_manual_basic_no_reference",
        feature_count=44,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=xjtu_sy split=xjtu_bearing_index_split "
            "feature=manual_basic label=degradation_three_tasks task=rul_linear_sequence model=gru "
            "trainer=base run.name=xjtu_main_rul_linear_gru_sequence_full_manual_basic_no_reference_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/xjtu "
            "'task.feature_columns.exclude_columns=[mag__time__rms]' trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
    SequenceRunPlan(
        experiment_id="xjtu_main_health_gru_sequence_compact_non_label_source_200ep",
        dataset="XJTU-SY",
        task="health_state_sequence",
        task_type="multiclass_classification",
        target="health_state_id",
        model="gru",
        feature_subset="compact_non_label_source",
        feature_count=6,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=xjtu_sy split=xjtu_bearing_index_split "
            "feature=manual_basic label=degradation_three_tasks task=health_state_sequence model=gru "
            "trainer=base run.name=xjtu_main_health_gru_sequence_compact_non_label_source_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/xjtu "
            "task.feature_columns.include=patterns "
            "'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,h__time__mean_abs,h__time__std,h__time__rms]' "
            "trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
    SequenceRunPlan(
        experiment_id="xjtu_main_early_gru_sequence_compact_non_label_source_200ep",
        dataset="XJTU-SY",
        task="early_fault_sequence",
        task_type="binary_classification",
        target="early_fault",
        model="gru",
        feature_subset="compact_non_label_source",
        feature_count=5,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=xjtu_sy split=xjtu_bearing_index_split "
            "feature=manual_basic label=degradation_three_tasks task=early_fault_sequence model=gru "
            "trainer=base run.name=xjtu_main_early_gru_sequence_compact_non_label_source_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/xjtu "
            "task.feature_columns.include=patterns "
            "'task.feature_columns.include_patterns=[mag__time__mean,mag__time__mean_abs,mag__time__std,v__time__std,v__time__mean_abs]' "
            "trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
    SequenceRunPlan(
        experiment_id="phm_official_rul_linear_gru_sequence_compact_non_label_source_200ep",
        dataset="PHM2012",
        task="rul_linear_sequence",
        task_type="regression",
        target="linear_rul_norm",
        model="gru",
        feature_subset="compact_non_label_source",
        feature_count=7,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=phm2012 split=phm2012_official "
            "feature=manual_basic label=degradation_three_tasks task=rul_linear_sequence model=gru "
            "trainer=base run.name=phm_official_rul_linear_gru_sequence_compact_non_label_source_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/phm2012 "
            "task.feature_columns.include=patterns "
            "'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__rms,h__time__std,v__time__mean_abs,mag__time__std]' "
            "trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
    SequenceRunPlan(
        experiment_id="phm_official_health_gru_sequence_compact_non_label_source_200ep",
        dataset="PHM2012",
        task="health_state_sequence",
        task_type="multiclass_classification",
        target="health_state_id",
        model="gru",
        feature_subset="compact_non_label_source",
        feature_count=5,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=phm2012 split=phm2012_official "
            "feature=manual_basic label=degradation_three_tasks task=health_state_sequence model=gru "
            "trainer=base run.name=phm_official_health_gru_sequence_compact_non_label_source_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/phm2012 "
            "task.feature_columns.include=patterns "
            "'task.feature_columns.include_patterns=[h__time__mean_abs,h__time__std,h__time__rms,mag__time__mean,mag__time__mean_abs]' "
            "trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
    SequenceRunPlan(
        experiment_id="phm_official_early_gru_sequence_compact_non_label_source_200ep",
        dataset="PHM2012",
        task="early_fault_sequence",
        task_type="binary_classification",
        target="early_fault",
        model="gru",
        feature_subset="compact_non_label_source",
        feature_count=7,
        sequence_length=8,
        label_source_included="no",
        command=(
            "uv run bp --config-name smoke mode=train dataset=phm2012 split=phm2012_official "
            "feature=manual_basic label=degradation_three_tasks task=early_fault_sequence model=gru "
            "trainer=base run.name=phm_official_early_gru_sequence_compact_non_label_source_200ep "
            "project.artifact_root=artifacts/baselines dataset.root=data/loader_roots/phm2012 "
            "task.feature_columns.include=patterns "
            "'task.feature_columns.include_patterns=[h__time__mean_abs,mag__time__mean,mag__time__mean_abs,h__time__std,h__time__rms,v__time__mean_abs,v__time__std]' "
            "trainer.batch_size=64 trainer.max_epochs=200"
        ),
    ),
]


def choose_primary_metric(task_type: str) -> Tuple[str, str]:
    if task_type == "regression":
        return "RMSE", "lower_is_better"
    if task_type in {"binary_classification", "multiclass_classification"}:
        return "WeightedF1", "higher_is_better"
    raise ValueError(f"Unsupported task_type: {task_type}")


def history_metric_columns(path: Path, primary_metric: str) -> Tuple[str, str, str]:
    history = load_json(path)
    if not history:
        raise ValueError(f"Empty history: {path}")
    val_primary = f"val_{primary_metric}"
    if val_primary not in history[0]:
        val_primary = "val_loss"
    return "train_loss", val_primary, "val_loss"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    for plan in AB_RUNS:
        raw_dir = latest_run_dir(args.artifact_root / "baselines" / "runs", plan.experiment_id)
        curated_dir = output / plan.experiment_id
        curate_run(plan, raw_dir, curated_dir)
        metrics_rows.append(metrics_row(plan, raw_dir))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output / "gru_sequence_200ep_metrics.csv", index=False)
    stale_comparison = output / "gru_sequence_50ep_vs_200ep.csv"
    if stale_comparison.exists():
        stale_comparison.unlink()
    write_summary_report(output, metrics)
    print(f"Sequence 200ep report written to {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Step AB GRU sequence outputs.")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/sequence_baseline_results"))
    return parser.parse_args(argv)


def curate_run(plan: SequenceRunPlan, raw_dir: Path, curated_dir: Path) -> None:
    if curated_dir.exists():
        shutil.rmtree(curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)
    (curated_dir / "command.txt").write_text(plan.command + "\n", encoding="utf-8")
    copy_file(raw_dir / "config" / "resolved.yaml", curated_dir / "resolved_config.yaml")
    copy_file(raw_dir / "task" / "task_spec.json", curated_dir / "task_spec.json")
    copy_file(raw_dir / "task" / "task_report.json", curated_dir / "task_report.json")
    copy_file(raw_dir / "task" / "feature_columns.txt", curated_dir / "feature_columns.txt")
    copy_file(raw_dir / "task" / "target_columns.txt", curated_dir / "target_columns.txt")
    copy_file(raw_dir / "metrics" / "history.json", curated_dir / "history.json")
    copy_file(raw_dir / "metrics" / "val_metrics.json", curated_dir / "val_metrics.json")
    copy_file(raw_dir / "metrics" / "test_metrics.json", curated_dir / "test_metrics.json")
    copy_file(raw_dir / "trainer" / "trainer_state.json", curated_dir / "trainer_state.json")
    copy_file(raw_dir / "trainer" / "model_summary.txt", curated_dir / "model_summary.txt")
    copy_file(raw_dir / "report.md", curated_dir / "experiment_report.md")

    figures = curated_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary_metric, _ = choose_primary_metric(plan.task_type)
    plot_training_curve(raw_dir / "metrics" / "history.json", figures / "training_curve.png", primary_metric)
    predictions = pd.read_parquet(raw_dir / "predictions" / "test_predictions.parquet")
    if plan.task_type == "regression":
        plot_true_pred_by_bearing(predictions, plan.target, figures / "test_true_pred_by_bearing.png")
        plot_pred_vs_true(predictions, plan.target, figures / "test_pred_vs_true.png")
        plot_residuals(predictions, plan.target, figures / "test_residuals.png")
    else:
        plot_confusion(predictions, figures / "test_confusion_matrix.png")
        plot_class_distribution(predictions, figures / "test_class_distribution.png")


def metrics_row(plan: SequenceRunPlan, raw_dir: Path) -> Dict[str, Any]:
    state = load_json(raw_dir / "trainer" / "trainer_state.json")
    history = load_json(raw_dir / "metrics" / "history.json")
    val_metrics = load_json(raw_dir / "metrics" / "val_metrics.json")
    test_metrics = load_json(raw_dir / "metrics" / "test_metrics.json")
    primary_metric, direction = choose_primary_metric(plan.task_type)
    last_epoch = int(state.get("epoch", 0))
    status = "completed" if last_epoch == 200 and len(history) == 200 else "incomplete"
    return {
        "experiment_id": plan.experiment_id,
        "dataset": plan.dataset,
        "task": plan.task,
        "task_type": plan.task_type,
        "target": plan.target,
        "model": plan.model,
        "feature_subset": plan.feature_subset,
        "feature_count": plan.feature_count,
        "sequence_length": plan.sequence_length,
        "label_source_included": plan.label_source_included,
        "max_epochs": 200,
        "last_epoch": last_epoch,
        "best_epoch": int(state.get("best_epoch", 0)),
        "primary_metric": primary_metric,
        "metric_direction": direction,
        "val_primary": float(val_metrics.get(primary_metric, math.nan)),
        "test_primary": float(test_metrics.get(primary_metric, math.nan)),
        "val_loss": float(val_metrics.get("loss", math.nan)),
        "test_loss": float(test_metrics.get("loss", math.nan)),
        "status": status,
        "notes": (
            "RUL uses linear_rul_norm; this is a 200ep GRU sequence run."
            if plan.task_type == "regression"
            else f"Classification target is {plan.target}; this is a 200ep GRU sequence run."
        ),
    }


def write_summary_report(output: Path, metrics: pd.DataFrame) -> None:
    lines = [
        "# Step AB: GRU Sequence 200ep Batch",
        "",
        "## 1. Purpose",
        "",
        "Run six 200-epoch GRU sequence models for XJTU-SY and PHM2012. RUL uses `linear_rul_norm`; classification tasks keep `health_state_id` and `early_fault`.",
        "",
        "## 2. Config",
        "",
        "- input_mode: `feature_sequence`",
        "- sequence.length: 8",
        "- model: `gru`",
        "- max_epochs: 200",
        "- batch_size: 64",
        "- label_source_included: no",
        "",
        "## 3. Experiments",
        "",
        markdown_table(metrics[[
            "experiment_id",
            "dataset",
            "task",
            "target",
            "feature_count",
            "status",
        ]]),
        "",
        "## 4. Training Completion",
        "",
        markdown_table(metrics[[
            "experiment_id",
            "last_epoch",
            "best_epoch",
            "primary_metric",
            "val_primary",
            "test_primary",
            "status",
        ]]),
        "",
        "## 5. Findings",
        "",
        "- This report intentionally evaluates only the six Step AB 200ep GRU sequence runs; older 50ep runs are not used for the current conclusion.",
        "- XJTU-SY RUL 200ep completed on `linear_rul_norm`; judge it from the 200ep metrics and RUL figures in this report.",
        "- XJTU-SY HealthState and EarlyFault 200ep both completed; EarlyFault is the strongest XJTU classification result in this batch.",
        "- PHM2012 RUL 200ep completed on `linear_rul_norm`; it is not directly comparable to old piecewise-RUL RMSE.",
        "- PHM2012 HealthState and EarlyFault completed; both need visual inspection through the generated confusion matrices.",
        "",
        "## 6. Figures",
        "",
        "- RUL directories contain `training_curve.png`, `test_true_pred_by_bearing.png`, `test_pred_vs_true.png`, and `test_residuals.png`.",
        "- Classification directories contain `training_curve.png`, `test_confusion_matrix.png`, and `test_class_distribution.png`.",
        "",
        "## 7. Decision",
        "",
        "- [x] Pass: six 200ep GRU sequence runs completed and were curated.",
        "- [ ] Needs rerun",
        "- [ ] Blocked",
        "",
    ]
    (output / "02_gru_sequence_200ep_batch.md").write_text("\n".join(lines), encoding="utf-8")


def plot_training_curve(history_path: Path, output_path: Path, primary_metric: str) -> None:
    history = pd.DataFrame(load_json(history_path))
    train_loss, val_primary, val_loss = history_metric_columns(history_path, primary_metric)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), dpi=160, sharex=True)
    axes[0].plot(history["epoch"], history[train_loss], label=train_loss, color="#1f77b4", linewidth=1.5)
    axes[0].plot(history["epoch"], history[val_loss], label=val_loss, color="#d62728", linewidth=1.5)
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history[val_primary], label=val_primary, color="#2ca02c", linewidth=1.5)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(val_primary)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_true_pred_by_bearing(predictions: pd.DataFrame, target: str, output_path: Path) -> None:
    frame = with_regression_columns(predictions, target)
    bearings = list(frame.groupby("bearing_id", dropna=False).groups.keys())
    rows = len(bearings)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(3.2, 3.1 * rows)), dpi=160, squeeze=False)
    for ax, (bearing_id, group) in zip(axes.flat, frame.groupby("bearing_id", dropna=False)):
        group = group.sort_values("target_timestep")
        rmse = float(math.sqrt(np.mean((group["y_pred"] - group["y_true"]) ** 2)))
        ax.plot(group["target_timestep"], group["y_true"], color="#1f77b4", linewidth=2.0, label="true")
        ax.plot(group["target_timestep"], group["y_pred"], color="#d62728", linewidth=1.5, label="pred")
        ax.set_title(f"{bearing_id} RMSE={rmse:.4f}")
        ax.set_xlabel("target_timestep")
        ax.set_ylabel(target)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true(predictions: pd.DataFrame, target: str, output_path: Path) -> None:
    frame = with_regression_columns(predictions, target)
    lower = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    upper = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=160)
    ax.scatter(frame["y_true"], frame["y_pred"], s=10, alpha=0.4)
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=1.2, label="ideal")
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    ax.set_title(f"Predicted vs true {target}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residuals(predictions: pd.DataFrame, target: str, output_path: Path) -> None:
    frame = with_regression_columns(predictions, target)
    residuals = frame["y_pred"] - frame["y_true"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
    ax.hist(residuals, bins=40, color="#4c78a8", alpha=0.85)
    ax.axvline(0.0, color="#d62728", linewidth=1.3)
    ax.set_xlabel("prediction residual")
    ax.set_ylabel("count")
    ax.set_title(f"Residuals for {target}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion(predictions: pd.DataFrame, output_path: Path) -> None:
    labels = sorted(set(predictions["y_true"].astype(int).tolist()) | set(predictions["y_pred"].astype(int).tolist()))
    matrix = confusion_matrix(predictions["y_true"].astype(int), predictions["y_pred"].astype(int), labels=labels)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=160)
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_class_distribution(predictions: pd.DataFrame, output_path: Path) -> None:
    true_counts = predictions["y_true"].astype(int).value_counts().sort_index()
    pred_counts = predictions["y_pred"].astype(int).value_counts().sort_index()
    labels = sorted(set(true_counts.index.tolist()) | set(pred_counts.index.tolist()))
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
    ax.bar(x - width / 2, [true_counts.get(label, 0) for label in labels], width, label="true")
    ax.bar(x + width / 2, [pred_counts.get(label, 0) for label in labels], width, label="pred")
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title("Test class distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def with_regression_columns(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    result = frame.copy()
    result["y_true"] = result[f"y_true__{target}"].astype(float)
    result["y_pred"] = result[f"y_pred__{target}"].astype(float)
    return result


def latest_run_dir(root: Path, experiment_id: str, required: bool = True) -> Path | None:
    candidates = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "trainer" / "trainer_state.json").exists():
            continue
        run_json = path / "run.json"
        if not run_json.exists():
            continue
        try:
            run_name = str(load_json(run_json).get("run_name", ""))
        except json.JSONDecodeError:
            continue
        if run_name == experiment_id:
            candidates.append(path)
    if not candidates:
        if required:
            raise FileNotFoundError(f"No complete raw run found for {experiment_id}")
        return None
    return candidates[-1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    rows = []
    for _, row in frame.iterrows():
        rows.append([format_cell(row[column]) for column in frame.columns])
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()

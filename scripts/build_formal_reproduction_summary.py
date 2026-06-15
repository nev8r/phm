"""
Build an aggregate formal reproduction summary from completed real-data runs.

The script does not train models. It indexes existing run outputs so the
formal validator can verify history, prediction, metric, attention and paper
reference files from one root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Build formal reproduction summary from existing outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cnn-root", type=Path, required=True)
    parser.add_argument("--xlstm-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cnn-max-samples", type=int, default=96)
    parser.add_argument("--xlstm-max-samples", type=int, default=96)
    return parser.parse_args()


def _split_entities(value: Any) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _run_dir_from_history(history_path: Path) -> Path:
    if history_path.name != "history.csv" or history_path.parent.parent.name != "experiments":
        raise ValueError(f"unexpected history path shape: {history_path}")
    return history_path.parent.parent.parent


def _row_metrics(row: pd.Series, names: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in names:
        if name in row and not pd.isna(row[name]):
            metrics[name] = float(row[name])
    return metrics


def _build_runs(frame: pd.DataFrame, *, metric_names: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        history_path = Path(str(row["history_path"]))
        run_dir = _run_dir_from_history(history_path)
        run: dict[str, Any] = {
            "dataset_name": row["dataset_name"],
            "condition_name": row["condition_name"],
            "model_name": row["model_name"],
            "data_source": row["data_source"],
            "train_entities": _split_entities(row.get("train_entities", "")),
            "test_entities": _split_entities(row.get("test_entities", "")),
            "prediction_count": int(row["prediction_count"]),
            "epoch_count": int(row["epoch_count"]),
            "history_path": str(history_path),
            "prediction_path": str(run_dir / "predictions.csv"),
            "metrics_path": str(run_dir / "metrics.json"),
            "attention_path": str(run_dir / "attention_weights.csv"),
            "metrics": _row_metrics(row, metric_names),
        }
        if "train_sequence_count" in row and not pd.isna(row["train_sequence_count"]):
            run["train_sequence_count"] = int(row["train_sequence_count"])
        if "test_sequence_count" in row and not pd.isna(row["test_sequence_count"]):
            run["test_sequence_count"] = int(row["test_sequence_count"])
        runs.append(run)
    return runs


def _section(
    *,
    paper: str,
    source: str,
    mode: str,
    root: Path,
    metric_names: list[str],
    used_condition_count: int | None = None,
    used_dataset_count: int | None = None,
) -> dict[str, Any]:
    comparison_path = root / "comparison_metrics.csv"
    paper_reference_path = root / "paper_reference_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"missing comparison CSV: {comparison_path}")
    if not paper_reference_path.exists():
        raise FileNotFoundError(f"missing paper reference CSV: {paper_reference_path}")
    frame = pd.read_csv(comparison_path)
    return {
        "paper": paper,
        "source": source,
        "mode": mode,
        "comparison_path": str(comparison_path),
        "paper_reference_path": str(paper_reference_path),
        "trained_model_count": int(len(frame)),
        "used_condition_count": used_condition_count,
        "used_dataset_count": used_dataset_count,
        "runs": _build_runs(frame, metric_names=metric_names),
    }


def build_summary(
    *,
    output_root: Path,
    cnn_root: Path,
    xlstm_root: Path,
    epochs: int,
    batch_size: int,
    cnn_max_samples: int,
    xlstm_max_samples: int,
) -> Path:
    """
    build and write the aggregate summary

    Returns
    -------
    Path
        written summary path
    """

    summary = {
        "status": "OK",
        "mode": "formal_real_data",
        "epoch_count": epochs,
        "batch_size": batch_size,
        "cnn_max_samples_per_entity": cnn_max_samples,
        "xlstm_max_samples_per_entity": xlstm_max_samples,
        "cnn_lstm_attention": _section(
            paper="Life prediction method of rolling bearing based on CNN-LSTM-AM",
            source="https://www.extrica.com/article/23793",
            mode="formal_real_data_split",
            root=cnn_root,
            metric_names=["rmse", "normalized_rmse", "huang_rul_score", "phm2012_score_scaled"],
            used_condition_count=2,
        ),
        "xlstm_transformer": _section(
            paper="An xLSTM-Transformer method for remaining useful life prediction",
            source="https://www.mdpi.com/1424-8220/26/5/1578",
            mode="formal_real_data_split",
            root=xlstm_root,
            metric_names=["rmse", "normalized_rmse", "r2_score", "phm2012_score", "huang_rul_score"],
            used_condition_count=6,
            used_dataset_count=2,
        ),
    }
    summary_path = output_root / "formal_paper_reproductions" / "formal_reproduction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    """
    command line entry point
    """

    args = parse_args()
    summary_path = build_summary(
        output_root=args.output_root,
        cnn_root=args.cnn_root,
        xlstm_root=args.xlstm_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cnn_max_samples=args.cnn_max_samples,
        xlstm_max_samples=args.xlstm_max_samples,
    )
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()

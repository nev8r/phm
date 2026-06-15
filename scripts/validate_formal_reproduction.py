"""
Validate formal RUL paper reproduction outputs.

The validator intentionally rejects demo-scale evidence. It checks the
aggregate summary, per-run outputs, comparison CSV columns, epoch counts,
prediction counts, and real-data source markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CORE_OUTPUT_FIELDS = ["history_path", "prediction_path", "metrics_path", "attention_path"]
REFERENCE_REQUIRED_COLUMNS = {
    "paper",
    "dataset_name",
    "model_name",
    "paper_metric_name",
    "local_metric_name",
    "paper_value",
    "local_value",
    "relative_gap_pct",
    "abs_relative_gap_pct",
    "pass_threshold_pct",
    "within_threshold",
}

CNN_REQUIRED_COLUMNS = {
    "dataset_name",
    "condition_name",
    "model_name",
    "rmse",
    "normalized_rmse",
    "smape",
    "huang_rul_score",
    "over_prediction_rate",
    "within_10_percent_rate",
    "prediction_count",
    "epoch_count",
    "train_sequence_count",
    "test_sequence_count",
}

XLSTM_REQUIRED_COLUMNS = {
    "dataset_name",
    "condition_name",
    "model_name",
    "rmse",
    "normalized_rmse",
    "r2",
    "r2_score",
    "phm2012_score",
    "huang_rul_score",
    "prediction_count",
    "epoch_count",
    "rmse_change_pct_vs_transformer",
    "score_change_pct_vs_transformer",
}


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Validate formal RUL reproduction evidence.")
    parser.add_argument("summary_or_root", type=Path)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--min-predictions", type=int, default=30)
    parser.add_argument(
        "--min-reference-pass-rate",
        type=float,
        default=0.0,
        help="Minimum aggregate paper-reference within-threshold pass rate; use 0.0 to only report it.",
    )
    return parser.parse_args()


def find_summary_path(summary_or_root: Path) -> Path:
    """
    find the aggregate formal reproduction summary path

    Parameters
    ----------
    summary_or_root : Path
        summary file or output root

    Returns
    -------
    Path
        summary file path
    """

    if summary_or_root.is_file():
        return summary_or_root
    matches = sorted(summary_or_root.rglob("formal_reproduction_summary.json"))
    if not matches:
        raise FileNotFoundError(f"formal_reproduction_summary.json not found under {summary_or_root}")
    if len(matches) > 1:
        return matches[-1]
    return matches[0]


def validate_summary(
    summary_path: Path,
    *,
    min_epochs: int,
    min_predictions: int,
    min_reference_pass_rate: float = 0.0,
) -> dict[str, Any]:
    """
    validate formal reproduction summary

    Parameters
    ----------
    summary_path : Path
        aggregate summary path
    min_epochs : int
        minimum epoch count per run
    min_predictions : int
        minimum prediction count per run

    Returns
    -------
    dict[str, Any]
        loaded summary
    """

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "OK":
        raise AssertionError(f"summary status is not OK: {summary.get('status')}")
    if summary.get("mode") != "formal_real_data":
        raise AssertionError(f"summary mode is not formal_real_data: {summary.get('mode')}")
    aggregate_reference_rows = 0
    aggregate_reference_passes = 0
    reference_summaries: dict[str, dict[str, Any]] = {}
    for section_name, required_columns in [
        ("cnn_lstm_attention", CNN_REQUIRED_COLUMNS),
        ("xlstm_transformer", XLSTM_REQUIRED_COLUMNS),
    ]:
        section = summary.get(section_name)
        if not isinstance(section, dict):
            raise AssertionError(f"missing section: {section_name}")
        _validate_comparison(section, required_columns=required_columns)
        reference_summary = _validate_paper_reference(section)
        reference_summaries[section_name] = reference_summary
        aggregate_reference_rows += int(reference_summary["row_count"])
        aggregate_reference_passes += int(reference_summary["pass_count"])
        _validate_runs(section, min_epochs=min_epochs, min_predictions=min_predictions)
    aggregate_pass_rate = aggregate_reference_passes / aggregate_reference_rows if aggregate_reference_rows else 0.0
    if aggregate_pass_rate < min_reference_pass_rate:
        raise AssertionError(
            "aggregate paper-reference pass rate is below "
            f"{min_reference_pass_rate:.3f}: {aggregate_pass_rate:.3f} "
            f"({aggregate_reference_passes}/{aggregate_reference_rows})"
        )
    summary["reference_validation"] = {
        "aggregate_pass_count": aggregate_reference_passes,
        "aggregate_row_count": aggregate_reference_rows,
        "aggregate_pass_rate": aggregate_pass_rate,
        "sections": reference_summaries,
    }
    return summary


def _validate_comparison(section: dict[str, Any], *, required_columns: set[str]) -> None:
    comparison_path = Path(str(section.get("comparison_path", "")))
    if not comparison_path.exists():
        raise AssertionError(f"missing comparison CSV: {comparison_path}")
    comparison_frame = pd.read_csv(comparison_path)
    missing_columns = required_columns.difference(comparison_frame.columns)
    if missing_columns:
        raise AssertionError(f"{comparison_path} missing columns: {sorted(missing_columns)}")
    if "generated_demo_files" in set(comparison_frame.get("data_source", [])):
        raise AssertionError(f"{comparison_path} contains generated_demo_files rows")


def _validate_paper_reference(section: dict[str, Any]) -> dict[str, Any]:
    reference_path = Path(str(section.get("paper_reference_path", "")))
    if not reference_path.exists():
        raise AssertionError(f"missing paper reference comparison CSV: {reference_path}")
    reference_frame = pd.read_csv(reference_path)
    missing_columns = REFERENCE_REQUIRED_COLUMNS.difference(reference_frame.columns)
    if missing_columns:
        raise AssertionError(f"{reference_path} missing columns: {sorted(missing_columns)}")
    if reference_frame.empty:
        raise AssertionError(f"{reference_path} is empty")
    if reference_frame["local_value"].isna().any():
        raise AssertionError(f"{reference_path} contains missing local_value entries")
    pass_count = int(reference_frame["within_threshold"].astype(bool).sum())
    row_count = int(len(reference_frame))
    return {
        "path": str(reference_path),
        "pass_count": pass_count,
        "row_count": row_count,
        "pass_rate": pass_count / row_count if row_count else 0.0,
    }


def _validate_runs(section: dict[str, Any], *, min_epochs: int, min_predictions: int) -> None:
    runs = section.get("runs")
    if not isinstance(runs, list) or not runs:
        raise AssertionError(f"{section.get('paper')} has no runs")
    for run in runs:
        if not isinstance(run, dict):
            raise AssertionError("run entry must be a dictionary")
        run_name = f"{run.get('dataset_name')} / {run.get('condition_name')} / {run.get('model_name')}"
        if run.get("data_source") != "real_or_provided_files":
            raise AssertionError(f"{run_name} is not real-data sourced: {run.get('data_source')}")
        if int(run.get("epoch_count", 0)) < min_epochs:
            raise AssertionError(f"{run_name} epoch_count is below {min_epochs}: {run.get('epoch_count')}")
        if int(run.get("prediction_count", 0)) < min_predictions:
            raise AssertionError(
                f"{run_name} prediction_count is below {min_predictions}: {run.get('prediction_count')}"
            )
        for field_name in CORE_OUTPUT_FIELDS:
            output_path = Path(str(run.get(field_name, "")))
            if not output_path.exists():
                raise AssertionError(f"{run_name} missing {field_name}: {output_path}")
        history_frame = pd.read_csv(Path(str(run["history_path"])))
        if len(history_frame) < min_epochs:
            raise AssertionError(f"{run_name} history rows below {min_epochs}: {len(history_frame)}")
        prediction_frame = pd.read_csv(Path(str(run["prediction_path"])))
        if len(prediction_frame) != int(run["prediction_count"]):
            raise AssertionError(f"{run_name} prediction_count does not match prediction CSV")


def main() -> None:
    """
    command line entry point
    """

    args = parse_args()
    summary_path = find_summary_path(args.summary_or_root)
    summary = validate_summary(
        summary_path,
        min_epochs=args.min_epochs,
        min_predictions=args.min_predictions,
        min_reference_pass_rate=args.min_reference_pass_rate,
    )
    reference_validation = summary["reference_validation"]
    print(f"validated {summary_path}")
    print(
        "paper_reference_pass_rate="
        f"{reference_validation['aggregate_pass_rate']:.3f} "
        f"({reference_validation['aggregate_pass_count']}/{reference_validation['aggregate_row_count']})"
    )


if __name__ == "__main__":
    main()

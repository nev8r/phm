"""
Formal paper reproduction validation tests.

this file is for testing anti-demo evidence gates

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_validator_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "validate_formal_reproduction.py"
    spec = importlib.util.spec_from_file_location("validate_formal_reproduction", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validate_formal_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_validator_accepts_complete_real_data_summary(tmp_path: Path) -> None:
    validator = _load_validator_module()
    summary_path = _write_fake_summary(tmp_path, epoch_count=50, prediction_count=35, data_source="real_or_provided_files")

    summary = validator.validate_summary(summary_path, min_epochs=50, min_predictions=30)

    assert summary["status"] == "OK"
    assert summary["reference_validation"]["aggregate_pass_rate"] == 1.0


def test_formal_validator_rejects_demo_source(tmp_path: Path) -> None:
    validator = _load_validator_module()
    summary_path = _write_fake_summary(tmp_path, epoch_count=50, prediction_count=35, data_source="generated_demo_files")

    with pytest.raises(AssertionError, match="generated_demo_files|real-data"):
        validator.validate_summary(summary_path, min_epochs=50, min_predictions=30)


def test_formal_validator_rejects_too_few_epochs(tmp_path: Path) -> None:
    validator = _load_validator_module()
    summary_path = _write_fake_summary(tmp_path, epoch_count=8, prediction_count=35, data_source="real_or_provided_files")

    with pytest.raises(AssertionError, match="epoch_count"):
        validator.validate_summary(summary_path, min_epochs=50, min_predictions=30)


def test_formal_validator_can_require_reference_pass_rate(tmp_path: Path) -> None:
    validator = _load_validator_module()
    summary_path = _write_fake_summary(
        tmp_path,
        epoch_count=50,
        prediction_count=35,
        data_source="real_or_provided_files",
        reference_pass=False,
    )

    with pytest.raises(AssertionError, match="paper-reference pass rate"):
        validator.validate_summary(
            summary_path,
            min_epochs=50,
            min_predictions=30,
            min_reference_pass_rate=0.5,
        )


def _write_fake_summary(
    tmp_path: Path,
    *,
    epoch_count: int,
    prediction_count: int,
    data_source: str,
    reference_pass: bool = True,
) -> Path:
    history_path = tmp_path / "history.csv"
    prediction_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"
    attention_path = tmp_path / "attention_weights.csv"
    pd.DataFrame({"epoch": list(range(1, epoch_count + 1)), "val_rmse": [1.0] * epoch_count}).to_csv(
        history_path,
        index=False,
    )
    pd.DataFrame({"target": list(range(prediction_count)), "prediction": list(range(prediction_count))}).to_csv(
        prediction_path,
        index=False,
    )
    metrics_path.write_text(json.dumps({"rmse": 0.0, "huang_rul_score": 0.0}), encoding="utf-8")
    pd.DataFrame({"attention_0": [1.0] * prediction_count}).to_csv(attention_path, index=False)

    cnn_comparison = tmp_path / "cnn_comparison_metrics.csv"
    xlstm_comparison = tmp_path / "xlstm_comparison_metrics.csv"
    cnn_reference = tmp_path / "cnn_paper_reference_comparison.csv"
    xlstm_reference = tmp_path / "xlstm_paper_reference_comparison.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "XJTU-SY",
                "condition_name": "condition_1",
                "model_name": "CNN-LSTM-AM",
                "data_source": data_source,
                "rmse": 1.0,
                "normalized_rmse": 0.1,
                "smape": 0.1,
                "huang_rul_score": 1.0,
                "over_prediction_rate": 0.0,
                "within_10_percent_rate": 1.0,
                "prediction_count": prediction_count,
                "epoch_count": epoch_count,
                "train_sequence_count": 40,
                "test_sequence_count": prediction_count,
            }
        ]
    ).to_csv(cnn_comparison, index=False)
    pd.DataFrame(
        [
            {
                "dataset_name": "XJTU-SY",
                "condition_name": "condition_1",
                "model_name": "XLSTM-Transformer",
                "data_source": data_source,
                "rmse": 1.0,
                "normalized_rmse": 0.1,
                "r2": 0.9,
                "r2_score": 0.9,
                "phm2012_score": 1.0,
                "huang_rul_score": 1.0,
                "prediction_count": prediction_count,
                "epoch_count": epoch_count,
                "rmse_change_pct_vs_transformer": 0.0,
                "score_change_pct_vs_transformer": 0.0,
            }
        ]
    ).to_csv(xlstm_comparison, index=False)
    _write_fake_reference(cnn_reference, within_threshold=reference_pass)
    _write_fake_reference(xlstm_reference, within_threshold=reference_pass)

    run_summary = {
        "dataset_name": "XJTU-SY",
        "condition_name": "condition_1",
        "model_name": "CNN-LSTM-AM",
        "data_source": data_source,
        "prediction_count": prediction_count,
        "epoch_count": epoch_count,
        "history_path": str(history_path),
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "attention_path": str(attention_path),
        "metrics": {"rmse": 1.0, "huang_rul_score": 1.0},
    }
    summary = {
        "status": "OK",
        "mode": "formal_real_data",
        "cnn_lstm_attention": {
            "paper": "cnn",
            "comparison_path": str(cnn_comparison),
            "paper_reference_path": str(cnn_reference),
            "runs": [run_summary],
        },
        "xlstm_transformer": {
            "paper": "xlstm",
            "comparison_path": str(xlstm_comparison),
            "paper_reference_path": str(xlstm_reference),
            "runs": [{**run_summary, "model_name": "XLSTM-Transformer"}],
        },
    }
    summary_path = tmp_path / "formal_reproduction_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _write_fake_reference(output_path: Path, *, within_threshold: bool) -> None:
    pd.DataFrame(
        [
            {
                "paper": "fake",
                "dataset_name": "XJTU-SY",
                "condition_name": "condition_1",
                "model_name": "CNN-LSTM-AM",
                "paper_metric_name": "rmse",
                "local_metric_name": "normalized_rmse",
                "paper_value": 0.1,
                "local_value": 0.1,
                "relative_gap_pct": 0.0 if within_threshold else 200.0,
                "abs_relative_gap_pct": 0.0 if within_threshold else 200.0,
                "pass_threshold_pct": 50.0,
                "within_threshold": within_threshold,
                "note": "fake reference row",
            }
        ]
    ).to_csv(output_path, index=False)

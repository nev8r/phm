"""
test prediction audit module.

Purpose: verify test prediction audit module behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import math

import pandas as pd

from recipes.diagnostics.audit_predictions import (
    _markdown_table,
    alignment_check_row,
    classification_per_class_rows,
    naive_baseline_rows,
    normalize_prediction_frame,
    rul_range_check_row,
)


def test_normalize_prediction_frame_accepts_mlp_and_sklearn_regression_columns():
    mlp_frame = pd.DataFrame({
        "sample_uid": ["s0"],
        "target_sample_uid": ["s0"],
        "split": ["test"],
        "bearing_id": ["b0"],
        "target_timestep": [0],
        "y_true__piecewise_rul_norm": [0.25],
        "y_pred__piecewise_rul_norm": [0.50],
    })
    sklearn_frame = pd.DataFrame({
        "sample_uid": ["s1"],
        "split": ["test"],
        "bearing_id": ["b0"],
        "target_timestep": [1],
        "y_true": [0.75],
        "y_pred": [0.80],
    })

    mlp = normalize_prediction_frame(mlp_frame, ["piecewise_rul_norm"])
    sklearn = normalize_prediction_frame(sklearn_frame, ["piecewise_rul_norm"])

    assert mlp["y_true"].tolist() == [0.25]
    assert mlp["y_pred"].tolist() == [0.50]
    assert sklearn["target_sample_uid"].tolist() == ["s1"]
    assert sklearn["y_true"].tolist() == [0.75]


def test_alignment_check_reports_missing_duplicate_and_mismatched_targets():
    predictions = pd.DataFrame({
        "sample_uid": ["s0", "s1", "s1", "missing"],
        "target_sample_uid": ["s0", "s1", "s1", "missing"],
        "split": ["test", "test", "test", "test"],
        "bearing_id": ["b0", "b0", "b0", "b1"],
        "target_timestep": [0, 1, 2, 0],
        "y_true": [0.25, 0.75, 0.50, 1.0],
        "y_pred": [0.2, 0.7, 0.4, 0.9],
    })
    labels = pd.DataFrame({
        "sample_uid": ["s0", "s1"],
        "piecewise_rul_norm": [0.25, 0.70],
    })

    row = alignment_check_row(
        experiment_id="toy",
        dataset="toyset",
        task="rul_tabular",
        task_type="regression",
        model_family="toy_model",
        split="test",
        predictions=predictions,
        labels=labels,
        target_column="piecewise_rul_norm",
        manifest_available=True,
    )

    assert row["num_prediction_rows"] == 4
    assert row["num_duplicate_sample_uid"] == 1
    assert row["num_missing_labels"] == 1
    assert row["num_mismatched_targets"] == 2
    assert row["alignment_ok"] == "no"


def test_rul_range_check_reports_clip_rate_and_clipped_rmse():
    predictions = pd.DataFrame({
        "y_true": [0.0, 0.5, 1.0],
        "y_pred": [-0.4, 0.5, 1.4],
    })

    row = rul_range_check_row(
        experiment_id="toy",
        dataset="toyset",
        task="rul_tabular",
        model_family="toy_model",
        split="test",
        predictions=predictions,
    )

    assert row["num_pred_lt_0"] == 1
    assert row["num_pred_gt_1"] == 1
    assert math.isclose(row["clip_rate"], 2 / 3)
    assert row["clipped_RMSE"] < row["raw_RMSE"]
    assert row["clip_improves_rmse"] == "yes"


def test_naive_baseline_rows_compare_model_against_train_mean_for_rul():
    train_predictions = pd.DataFrame({"y_true": [0.0, 0.5, 1.0]})
    test_predictions = pd.DataFrame({"y_true": [0.0, 1.0], "y_pred": [0.1, 0.9]})

    rows = naive_baseline_rows(
        experiment_id="toy",
        dataset="toyset",
        task="rul_tabular",
        task_type="regression",
        model_family="toy_model",
        train_predictions=train_predictions,
        test_predictions=test_predictions,
    )

    by_strategy = {row["naive_strategy"]: row for row in rows}
    assert by_strategy["train_mean"]["primary_metric"] == "RMSE"
    assert by_strategy["train_mean"]["model_beats_naive"] == "yes"
    assert by_strategy["constant_zero"]["model_beats_naive"] == "yes"


def test_classification_per_class_rows_include_precision_recall_support():
    predictions = pd.DataFrame({
        "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 1, 1],
    })

    rows = classification_per_class_rows(
        experiment_id="toy",
        dataset="toyset",
        task="early_fault_tabular",
        model_family="toy_model",
        split="test",
        predictions=predictions,
    )

    by_class = {row["class_id"]: row for row in rows}
    assert by_class[0]["support"] == 2
    assert math.isclose(by_class[0]["recall"], 0.5)
    assert by_class[1]["support"] == 2
    assert math.isclose(by_class[1]["precision"], 2 / 3)


def test_markdown_table_does_not_require_optional_tabulate_dependency():
    table = _markdown_table(pd.DataFrame({"name": ["a"], "value": [0.1234567]}))

    assert "| name | value |" in table
    assert "| a | 0.123457 |" in table

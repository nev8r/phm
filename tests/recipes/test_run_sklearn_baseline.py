import math

import pandas as pd

from recipes.baselines.run_sklearn_baseline import (
    build_model,
    compute_metrics,
    dataset_to_arrays,
    expected_figure_files,
    feature_importance_frame,
    summarize_metric_gaps,
)
from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset


def _toy_dataset(task_type="regression"):
    features = pd.DataFrame({
        "sample_uid": ["s0", "s1"],
        "dataset": ["toy", "toy"],
        "bearing_id": ["b0", "b0"],
        "condition_id": ["c0", "c0"],
        "source_group": [None, None],
        "sample_id": ["000000", "000001"],
        "timestep": [0, 1],
        "f1": [1.0, 2.0],
        "f2": [3.0, 4.0],
    })
    labels = features[[
        "sample_uid",
        "dataset",
        "bearing_id",
        "condition_id",
        "source_group",
        "sample_id",
        "timestep",
    ]].copy()
    target = "piecewise_rul_norm" if task_type == "regression" else "early_fault"
    labels[target] = [0.25, 0.75] if task_type == "regression" else [0, 1]
    manifest = pd.DataFrame({
        "example_uid": ["train::b0::000000", "train::b0::000001"],
        "split": ["train", "train"],
        "dataset": ["toy", "toy"],
        "bearing_id": ["b0", "b0"],
        "condition_id": ["c0", "c0"],
        "source_group": [None, None],
        "start_sample_uid": ["s0", "s1"],
        "end_sample_uid": ["s0", "s1"],
        "target_sample_uid": ["s0", "s1"],
        "start_timestep": [0, 1],
        "end_timestep": [0, 1],
        "target_timestep": [0, 1],
        "num_timesteps": [1, 1],
        "window_sample_uids": ["s0", "s1"],
    })
    return TaskDataset(features, labels, manifest, ["f1", "f2"], [target], "tabular", task_type)


def test_build_model_uses_fixed_step_y_parameters():
    xgb = build_model("xgboost_regressor", "regression", random_state=42)
    rf = build_model("random_forest_classifier", "binary_classification", random_state=42)

    assert xgb.n_estimators == 300
    assert xgb.max_depth == 3
    assert xgb.learning_rate == 0.05
    assert xgb.objective == "reg:squarederror"
    assert xgb.tree_method == "hist"
    assert rf.n_estimators == 300
    assert rf.min_samples_leaf == 2
    assert rf.class_weight == "balanced"


def test_compute_metrics_matches_step_y_primary_rules():
    regression = compute_metrics("regression", [1.0, 3.0], [2.0, 1.0])
    classification = compute_metrics("binary_classification", [0, 1, 1], [0, 0, 1])

    assert regression["primary_metric"] == "RMSE"
    assert regression["metric_direction"] == "lower_is_better"
    assert math.isclose(regression["metrics"]["MAE"], 1.5)
    assert math.isclose(regression["metrics"]["MSE"], 2.5)
    assert math.isclose(regression["metrics"]["RMSE"], math.sqrt(2.5))
    assert classification["primary_metric"] == "WeightedF1"
    assert classification["metric_direction"] == "higher_is_better"
    assert "Accuracy" in classification["metrics"]
    assert "MacroF1" in classification["metrics"]
    assert "WeightedF1" in classification["metrics"]


def test_dataset_to_arrays_preserves_manifest_metadata():
    dataset = _toy_dataset()

    arrays = dataset_to_arrays(dataset)

    assert arrays.x.shape == (2, 2)
    assert arrays.y.tolist() == [0.25, 0.75]
    assert arrays.metadata["example_uid"].tolist() == ["train::b0::000000", "train::b0::000001"]
    assert arrays.metadata["sample_uid"].tolist() == ["s0", "s1"]


def test_feature_importance_frame_is_ranked_descending():
    frame = feature_importance_frame(["f1", "f2"], [0.2, 0.8])

    assert frame["feature"].tolist() == ["f2", "f1"]
    assert frame["rank"].tolist() == [1, 2]


def test_summarize_metric_gaps_respects_metric_direction():
    regression = summarize_metric_gaps(
        primary_metric="RMSE",
        metric_direction="lower_is_better",
        train_primary=0.10,
        val_primary=0.30,
        test_primary=0.40,
    )
    classification = summarize_metric_gaps(
        primary_metric="WeightedF1",
        metric_direction="higher_is_better",
        train_primary=0.90,
        val_primary=0.70,
        test_primary=0.65,
    )

    assert math.isclose(regression["train_val_gap"], 0.20)
    assert math.isclose(regression["val_test_gap"], 0.10)
    assert math.isclose(classification["train_val_gap"], 0.20)
    assert math.isclose(classification["val_test_gap"], 0.05)
    assert regression["gap_pattern"] == "train_best_test_worst"
    assert classification["gap_pattern"] == "train_best_test_worse"


def test_expected_figure_files_match_task_type_visual_checks():
    assert expected_figure_files("regression") == [
        "figures/train_pred_vs_true.png",
        "figures/val_pred_vs_true.png",
        "figures/test_pred_vs_true.png",
        "figures/test_residuals.png",
        "figures/feature_importance_top10.png",
    ]
    assert expected_figure_files("binary_classification") == [
        "figures/train_confusion_matrix.png",
        "figures/val_confusion_matrix.png",
        "figures/test_confusion_matrix.png",
        "figures/test_class_distribution.png",
        "figures/feature_importance_top10.png",
    ]

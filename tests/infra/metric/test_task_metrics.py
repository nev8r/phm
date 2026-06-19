"""
Test Stage 5 task metrics.
"""

import numpy as np

from USTC.SSE.BearingPrediction.infra.metric.TaskMetrics import classification_metrics, regression_metrics


def test_regression_metrics_compute_mae_mse_rmse():
    metrics = regression_metrics(np.array([[1.0], [0.0]]), np.array([[0.5], [0.0]]))

    assert metrics["MAE"] == 0.25
    assert metrics["MSE"] == 0.125
    assert round(metrics["RMSE"], 6) == round(0.125 ** 0.5, 6)


def test_classification_metrics_compute_accuracy_and_f1():
    metrics = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))

    assert metrics["Accuracy"] == 2 / 3
    assert "MacroF1" in metrics
    assert "WeightedF1" in metrics

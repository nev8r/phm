"""
Test Stage 5 prediction storage.

Purpose: verify test stage 5 prediction storage behavior
Author: zy
Program date: 2026-06
Copyright: USTC

2026
"""

import pandas as pd

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager
from USTC.SSE.BearingPrediction.infra.predict.PredictionStore import PredictionStore


def test_prediction_store_writes_regression_predictions(tmp_path):
    predictions = pd.DataFrame({
        "example_uid": ["e0"],
        "split": ["val"],
        "sample_uid": ["s0"],
        "target_sample_uid": ["s0"],
        "dataset": ["XJTU-SY"],
        "bearing_id": ["Bearing1_1"],
        "condition_id": ["35Hz12kN"],
        "target_timestep": [0],
        "y_true__piecewise_rul_norm": [1.0],
        "y_pred__piecewise_rul_norm": [0.8],
        "abs_error__piecewise_rul_norm": [0.2],
    })

    PredictionStore(ArtifactManager(tmp_path), write_csv=True).save("val", predictions)

    assert (tmp_path / "predictions" / "val_predictions.parquet").exists()
    assert (tmp_path / "predictions" / "val_predictions.csv").exists()


def test_prediction_store_writes_classification_predictions(tmp_path):
    predictions = pd.DataFrame({
        "example_uid": ["e0"],
        "split": ["test"],
        "sample_uid": ["s0"],
        "target_sample_uid": ["s0"],
        "dataset": ["XJTU-SY"],
        "bearing_id": ["Bearing1_1"],
        "condition_id": ["35Hz12kN"],
        "target_timestep": [0],
        "y_true": [1],
        "y_pred": [0],
        "prob__0": [0.6],
        "prob__1": [0.4],
    })

    PredictionStore(ArtifactManager(tmp_path), write_csv=False).save("test", predictions)

    saved = pd.read_parquet(tmp_path / "predictions" / "test_predictions.parquet")
    assert "prob__0" in saved.columns
    assert "y_pred" in saved.columns

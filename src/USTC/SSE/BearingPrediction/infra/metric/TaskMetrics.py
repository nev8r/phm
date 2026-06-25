"""
Task-level metrics from prediction arrays.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "WeightedF1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

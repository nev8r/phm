"""
Prediction frame builders.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import List

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, REGRESSION


PREDICTION_METADATA_COLUMNS = [
    "example_uid",
    "split",
    "sample_uid",
    "target_sample_uid",
    "dataset",
    "bearing_id",
    "condition_id",
    "target_timestep",
]


def build_prediction_frame(
        metadata: List[dict],
        y_true,
        raw_output,
        task_type: str,
        target_columns: List[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(metadata)
    if frame.empty:
        return frame
    y_true = np.asarray(y_true)
    raw_output = np.asarray(raw_output)
    if task_type == REGRESSION:
        return _regression_frame(frame, y_true, raw_output, target_columns)
    if task_type in CLASSIFICATION_TYPES:
        return _classification_frame(frame, y_true, raw_output)
    raise ValueError(f"Unsupported task_type: {task_type}")


def _regression_frame(frame: pd.DataFrame, y_true, y_pred, target_columns: List[str]) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(len(frame), len(target_columns))
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(frame), len(target_columns))
    for index, column in enumerate(target_columns):
        frame[f"y_true__{column}"] = y_true[:, index]
        frame[f"y_pred__{column}"] = y_pred[:, index]
        frame[f"abs_error__{column}"] = np.abs(y_pred[:, index] - y_true[:, index])
    return frame


def _classification_frame(frame: pd.DataFrame, y_true, logits) -> pd.DataFrame:
    y_true = np.asarray(y_true).reshape(-1)
    logits = np.asarray(logits, dtype=float)
    probabilities = _softmax(logits)
    frame["y_true"] = y_true.astype(int)
    frame["y_pred"] = probabilities.argmax(axis=1).astype(int)
    for class_index in range(probabilities.shape[1]):
        frame[f"prob__{class_index}"] = probabilities[:, class_index]
    return frame


def _softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)

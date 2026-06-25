"""
Task registry helpers.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.infra.task.types import (
    BINARY_CLASSIFICATION,
    FEATURE_SEQUENCE,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    TABULAR,
)


TASK_TYPE_ALIASES = {
    "regression": REGRESSION,
    "binary_classification": BINARY_CLASSIFICATION,
    "multiclass_classification": MULTICLASS_CLASSIFICATION,
}

INPUT_MODE_ALIASES = {
    "tabular": TABULAR,
    "feature_sequence": FEATURE_SEQUENCE,
}


def normalize_task_type(value: str) -> str:
    if value not in TASK_TYPE_ALIASES:
        raise ValueError(f"Unsupported task_type: {value}")
    return TASK_TYPE_ALIASES[value]


def normalize_input_mode(value: str) -> str:
    if value not in INPUT_MODE_ALIASES:
        raise ValueError(f"Unsupported input_mode: {value}")
    return INPUT_MODE_ALIASES[value]

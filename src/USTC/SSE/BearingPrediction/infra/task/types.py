"""
Task type constants.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

REGRESSION = "regression"
BINARY_CLASSIFICATION = "binary_classification"
MULTICLASS_CLASSIFICATION = "multiclass_classification"

TABULAR = "tabular"
FEATURE_SEQUENCE = "feature_sequence"

CLASSIFICATION_TYPES = {BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION}
INPUT_MODES = {TABULAR, FEATURE_SEQUENCE}
TASK_TYPES = {REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION}

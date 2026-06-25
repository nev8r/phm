"""
Task data validation helpers.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Iterable, List

import pandas as pd


class TaskValidator:
    def validate_sample_alignment(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        _ensure_unique(features, "features")
        _ensure_unique(labels, "labels")
        feature_uids = set(features["sample_uid"])
        label_uids = set(labels["sample_uid"])
        if feature_uids != label_uids:
            missing_labels = sorted(feature_uids - label_uids)
            missing_features = sorted(label_uids - feature_uids)
            raise ValueError(
                "features and labels must contain the same sample_uid set; "
                f"missing_labels={missing_labels[:5]}, missing_features={missing_features[:5]}"
            )

    def validate_target_columns(self, labels: pd.DataFrame, target_columns: Iterable[str]) -> None:
        missing = [column for column in target_columns if column not in labels.columns]
        if missing:
            raise ValueError(f"Missing target columns: {missing}")

    def validate_feature_columns(self, feature_columns: List[str]) -> None:
        if not feature_columns:
            raise ValueError("Task must use at least one feature column")


def _ensure_unique(frame: pd.DataFrame, name: str) -> None:
    if "sample_uid" not in frame.columns:
        raise ValueError(f"{name} must contain sample_uid")
    if frame["sample_uid"].duplicated().any():
        raise ValueError(f"{name}.sample_uid values must be unique")

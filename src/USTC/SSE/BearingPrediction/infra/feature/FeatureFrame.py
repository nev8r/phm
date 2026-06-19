"""
Feature frame schema.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


FEATURE_INDEX_COLUMNS = [
    "sample_uid",
    "dataset",
    "bearing_id",
    "condition_id",
    "source_group",
    "sample_id",
    "timestep",
]


@dataclass
class FeatureFrame:
    data: pd.DataFrame
    index_columns: List[str]
    feature_columns: List[str]
    backend_name: str
    feature_set_name: str
    spec: Dict

    def validate(self) -> None:
        missing = [column for column in self.index_columns if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing feature index columns: {missing}")
        if not self.feature_columns:
            raise ValueError("FeatureFrame must contain at least one feature column")
        if self.data["sample_uid"].duplicated().any():
            raise ValueError("FeatureFrame sample_uid values must be unique")
        values = self.data[self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("FeatureFrame contains NaN or Inf feature values")

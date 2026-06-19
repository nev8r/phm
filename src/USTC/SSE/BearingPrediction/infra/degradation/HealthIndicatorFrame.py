"""
Health indicator frame schema.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


HI_INDEX_COLUMNS = [
    "sample_uid",
    "dataset",
    "bearing_id",
    "condition_id",
    "source_group",
    "sample_id",
    "timestep",
]


@dataclass
class HealthIndicatorFrame:
    data: pd.DataFrame
    index_columns: List[str]
    hi_column: str
    spec: Dict

    def validate(self) -> None:
        missing = [column for column in self.index_columns if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing HI index columns: {missing}")
        for column in ["hi_raw", "hi_smooth", "hi_norm", "hi_source_column"]:
            if column not in self.data.columns:
                raise ValueError(f"Missing HI column: {column}")
        if self.data["sample_uid"].duplicated().any():
            raise ValueError("HealthIndicatorFrame sample_uid values must be unique")
        values = self.data[["hi_raw", "hi_smooth", "hi_norm"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("HealthIndicatorFrame contains NaN or Inf values")

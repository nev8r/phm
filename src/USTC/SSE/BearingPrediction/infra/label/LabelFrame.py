"""
Label frame schema.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


LABEL_INDEX_COLUMNS = [
    "sample_uid",
    "dataset",
    "bearing_id",
    "condition_id",
    "source_group",
    "sample_id",
    "timestep",
]


@dataclass
class LabelFrame:
    data: pd.DataFrame
    index_columns: List[str]
    label_columns: List[str]
    spec: Dict

    def validate(self) -> None:
        missing = [column for column in self.index_columns if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing label index columns: {missing}")
        if not self.label_columns:
            raise ValueError("LabelFrame must contain at least one label column")
        if self.data["sample_uid"].duplicated().any():
            raise ValueError("LabelFrame sample_uid values must be unique")
        numeric_columns = [
            column for column in self.label_columns
            if pd.api.types.is_numeric_dtype(self.data[column])
        ]
        if numeric_columns:
            values = self.data[numeric_columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("LabelFrame contains NaN or Inf numeric label values")

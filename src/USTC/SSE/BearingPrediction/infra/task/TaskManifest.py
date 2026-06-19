"""
Task manifest schema.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd


TASK_MANIFEST_COLUMNS = [
    "example_uid",
    "split",
    "dataset",
    "bearing_id",
    "condition_id",
    "source_group",
    "start_sample_uid",
    "end_sample_uid",
    "target_sample_uid",
    "start_timestep",
    "end_timestep",
    "target_timestep",
    "num_timesteps",
    "window_sample_uids",
]


@dataclass
class TaskManifest:
    data: pd.DataFrame
    columns: List[str]

    def validate(self) -> None:
        missing = [column for column in self.columns if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing task manifest columns: {missing}")
        if self.data.empty:
            raise ValueError("No task examples generated")
        if self.data["example_uid"].duplicated().any():
            raise ValueError("Task manifest example_uid values must be unique")
        if (self.data["num_timesteps"].astype(int) <= 0).any():
            raise ValueError("Task manifest num_timesteps must be positive")

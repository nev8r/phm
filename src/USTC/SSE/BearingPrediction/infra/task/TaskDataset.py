"""
PyTorch-style task dataset.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, FEATURE_SEQUENCE, REGRESSION, TABULAR


class TaskDataset(Dataset):
    def __init__(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            manifest: pd.DataFrame,
            feature_columns: List[str],
            target_columns: List[str],
            input_mode: str,
            task_type: str,
    ):
        self.features = features.set_index("sample_uid", drop=False)
        self.labels = labels.set_index("sample_uid", drop=False)
        self.manifest = manifest.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.input_mode = input_mode
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict:
        row = self.manifest.iloc[index]
        if self.input_mode == TABULAR:
            x = self._tabular_x(row)
        elif self.input_mode == FEATURE_SEQUENCE:
            x = self._sequence_x(row)
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        y = self._target_y(str(row["target_sample_uid"]))
        return {
            "x": x,
            "y": y,
            "sample_uid": str(row["target_sample_uid"]),
            "target_sample_uid": str(row["target_sample_uid"]),
            "example_uid": str(row["example_uid"]),
            "split": str(row["split"]),
            "dataset": str(row["dataset"]),
            "bearing_id": str(row["bearing_id"]),
            "condition_id": str(row["condition_id"]),
            "timestep": int(row["target_timestep"]),
            "target_timestep": int(row["target_timestep"]),
        }

    def _tabular_x(self, row) -> torch.Tensor:
        sample_uid = str(row["target_sample_uid"])
        values = self.features.loc[sample_uid, self.feature_columns].to_numpy(dtype="float32")
        return torch.tensor(values, dtype=torch.float32)

    def _sequence_x(self, row) -> torch.Tensor:
        sample_uids = str(row["window_sample_uids"]).split("|")
        values = self.features.loc[sample_uids, self.feature_columns].to_numpy(dtype="float32")
        return torch.tensor(values, dtype=torch.float32)

    def _target_y(self, sample_uid: str) -> torch.Tensor:
        values = self.labels.loc[sample_uid, self.target_columns]
        if self.task_type in CLASSIFICATION_TYPES:
            if len(self.target_columns) != 1:
                raise ValueError("Classification tasks support exactly one target column in Stage 4")
            return torch.tensor(int(values.iloc[0] if hasattr(values, "iloc") else values), dtype=torch.long)
        if self.task_type == REGRESSION:
            array = values.to_numpy(dtype="float32") if hasattr(values, "to_numpy") else [float(values)]
            return torch.tensor(array, dtype=torch.float32)
        raise ValueError(f"Unsupported task_type: {self.task_type}")

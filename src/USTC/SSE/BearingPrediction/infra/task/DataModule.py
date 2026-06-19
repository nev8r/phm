"""
Task data module container.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import DataLoader

from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset


@dataclass
class DataModule:
    train: Optional[TaskDataset]
    val: Optional[TaskDataset]
    test: Optional[TaskDataset]
    all: Optional[TaskDataset]
    task_manifest: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    task_spec: Dict
    task_report: Dict

    @property
    def input_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def output_dim(self) -> int:
        return len(self.target_columns)

    def splits(self) -> Dict[str, TaskDataset]:
        return {
            name: dataset for name, dataset in {
                "train": self.train,
                "val": self.val,
                "test": self.test,
                "all": self.all,
            }.items()
            if dataset is not None
        }

    def to_dataloader(self, split: str, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        datasets = self.splits()
        if split not in datasets:
            raise ValueError(f"Unknown or empty split: {split}")
        return DataLoader(datasets[split], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

"""
Data entity module

this file is for defining bearing entity and window dataset abstractions

created by cyy

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from USTC.SSE.BearingPrediction.common.serialization import ArtifactSerializer


@dataclass
class BearingEntity:
    """
    Time ordered bearing measurement entity.

    Parameters
    ----------
    entity_id : str
        bearing identifier
    dataset_name : str
        source dataset name
    samples : pd.DataFrame
        snapshot dataframe
    sample_rate : float
        signal sample rate
    metadata : dict[str, Any]
        entity metadata
    """

    entity_id: str
    dataset_name: str
    samples: pd.DataFrame
    sample_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def channel_names(self) -> list[str]:
        """
        list available signal channel names

        Returns
        -------
        list[str]
            available channels
        """

        reserved_columns = {"sample_index", "rul", "timestamp"}
        available_channels: list[str] = []
        for column_name in self.samples.columns:
            if column_name in reserved_columns:
                continue
            sample_values = self.samples[column_name].dropna()
            if sample_values.empty:
                continue
            first_value = sample_values.iloc[0]
            if isinstance(first_value, (np.ndarray, list, tuple)):
                available_channels.append(column_name)
        return available_channels

    def get_channel(self, channel_name: str) -> list[np.ndarray]:
        """
        get a list of signal arrays for one channel

        Parameters
        ----------
        channel_name : str
            channel name

        Returns
        -------
        list[np.ndarray]
            signal snapshots
        """

        if channel_name not in self.samples.columns:
            available = ", ".join(self.channel_names())
            raise KeyError(f"{channel_name} is not available in {self.entity_id}. available: {available}")
        return [np.asarray(signal_values, dtype=float) for signal_values in self.samples[channel_name]]

    def export(self, output_path: Path) -> None:
        """
        export entity samples to csv or pickle

        Parameters
        ----------
        output_path : Path
            export path
        """

        export_frame = self.samples.copy()
        ArtifactSerializer.save_dataframe(export_frame, output_path)


class BearingWindowDataset(Dataset):
    """
    Torch compatible bearing dataset with metadata and serialization helpers.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        metadata_frame: pd.DataFrame,
        *,
        task_type: str,
        target_name: str,
        input_name: str,
        feature_frame: pd.DataFrame | None = None,
        extra_targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.targets = np.asarray(targets)
        self.metadata_frame = metadata_frame.reset_index(drop=True).copy()
        self.task_type = task_type
        self.target_name = target_name
        self.input_name = input_name
        self.feature_frame = feature_frame.reset_index(drop=True).copy() if feature_frame is not None else None
        self.extra_targets = extra_targets or {}

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_dtype = torch.long if self.task_type == "classification" else torch.float32
        item = {
            "inputs": torch.tensor(self.inputs[index], dtype=torch.float32),
            "targets": torch.tensor(self.targets[index], dtype=target_dtype),
            "index": index,
        }
        if self.feature_frame is not None:
            item["features"] = torch.tensor(self.feature_frame.iloc[index].to_numpy(dtype=np.float32), dtype=torch.float32)
        for key, values in self.extra_targets.items():
            item[key] = torch.tensor(values[index])
        return item

    def split_by_ratio(self, train_ratio: float) -> tuple["BearingWindowDataset", "BearingWindowDataset"]:
        """
        split dataset by chronological ratio

        Parameters
        ----------
        train_ratio : float
            train split ratio

        Returns
        -------
        tuple[BearingWindowDataset, BearingWindowDataset]
            train and test datasets
        """

        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")

        split_index = max(1, min(int(len(self) * train_ratio), len(self) - 1))
        return self._slice(0, split_index), self._slice(split_index, len(self))

    def _slice(self, start_index: int, end_index: int) -> "BearingWindowDataset":
        """
        slice dataset rows

        Parameters
        ----------
        start_index : int
            start index
        end_index : int
            end index

        Returns
        -------
        BearingWindowDataset
            sliced dataset
        """

        feature_frame = self.feature_frame.iloc[start_index:end_index].reset_index(drop=True) if self.feature_frame is not None else None
        extra_targets = {
            key: np.asarray(values[start_index:end_index])
            for key, values in self.extra_targets.items()
        }
        return BearingWindowDataset(
            inputs=self.inputs[start_index:end_index],
            targets=self.targets[start_index:end_index],
            metadata_frame=self.metadata_frame.iloc[start_index:end_index].reset_index(drop=True),
            task_type=self.task_type,
            target_name=self.target_name,
            input_name=self.input_name,
            feature_frame=feature_frame,
            extra_targets=extra_targets,
        )

    def as_frame(self) -> pd.DataFrame:
        """
        export dataset into a tabular representation

        Returns
        -------
        pd.DataFrame
            tabular dataset
        """

        export_frame = self.metadata_frame.copy()
        export_frame[self.target_name] = self.targets
        export_frame["inputs"] = [row.tolist() for row in self.inputs]
        if self.feature_frame is not None:
            for column_name in self.feature_frame.columns:
                export_frame[column_name] = self.feature_frame[column_name].to_numpy()
        for key, values in self.extra_targets.items():
            export_frame[key] = values
        return export_frame

    def export(self, output_path: Path) -> None:
        """
        export dataset to csv or pickle

        Parameters
        ----------
        output_path : Path
            target path
        """

        ArtifactSerializer.save_dataframe(self.as_frame(), output_path)


class SyntheticBearingFactory:
    """
    Create synthetic bearing entities for tests and demos.
    """

    def __init__(self, random_state: int = 42, sample_rate: float = 25600.0) -> None:
        self.random_generator = np.random.default_rng(random_state)
        self.sample_rate = sample_rate

    def create_run_to_failure_entity(
        self,
        entity_id: str,
        *,
        snapshot_count: int = 48,
        signal_length: int = 2048,
        dataset_name: str = "Synthetic",
    ) -> BearingEntity:
        """
        create a synthetic run-to-failure entity

        Parameters
        ----------
        entity_id : str
            bearing identifier
        snapshot_count : int
            number of snapshots
        signal_length : int
            length of each signal snapshot
        dataset_name : str
            dataset name

        Returns
        -------
        BearingEntity
            synthetic entity
        """

        time_axis = np.linspace(0.0, 1.0, signal_length, endpoint=False)
        records: list[dict[str, Any]] = []
        base_frequency = float(self.random_generator.uniform(6.0, 14.0))

        for sample_index in range(snapshot_count):
            degradation_ratio = sample_index / max(snapshot_count - 1, 1)
            amplitude = 0.7 + (2.8 * degradation_ratio)
            modulation = base_frequency + (5.0 * degradation_ratio)
            horizontal_signal = (
                amplitude * np.sin(2.0 * np.pi * modulation * time_axis)
                + 0.25 * np.sin(2.0 * np.pi * (2.2 * modulation) * time_axis)
                + self.random_generator.normal(0.0, 0.12 + (0.3 * degradation_ratio), size=time_axis.size)
            )
            vertical_signal = (
                (0.6 + degradation_ratio) * np.cos(2.0 * np.pi * (modulation / 2.0) * time_axis)
                + self.random_generator.normal(0.0, 0.1 + (0.22 * degradation_ratio), size=time_axis.size)
            )
            impulse_mask = self.random_generator.random(time_axis.size) < (0.006 + (0.04 * degradation_ratio))
            horizontal_signal = horizontal_signal + impulse_mask.astype(float) * self.random_generator.normal(
                1.5 * degradation_ratio,
                0.2,
                size=time_axis.size,
            )
            records.append(
                {
                    "sample_index": sample_index,
                    "timestamp": sample_index,
                    "rul": snapshot_count - sample_index - 1,
                    "Horizontal Vibration": horizontal_signal.astype(np.float32),
                    "Vertical Vibration": vertical_signal.astype(np.float32),
                }
            )

        sample_frame = pd.DataFrame.from_records(records)
        return BearingEntity(
            entity_id=entity_id,
            dataset_name=dataset_name,
            samples=sample_frame,
            sample_rate=self.sample_rate,
            metadata={"synthetic": True, "snapshot_count": snapshot_count, "signal_length": signal_length},
        )

"""
PHM2012 dataset loader module

this file is for loading IEEE PHM 2012 bearing dataset entities

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.dataset.base import BaseBearingLoader, DatasetResource


class PHM2012Loader(BaseBearingLoader):
    """
    Loader for the IEEE PHM 2012 FEMTO bearing dataset.
    """

    known_split_names = {"Learning_set", "Test_set", "Full_Test_Set"}
    condition_metadata = {
        "1": {"operating_condition": "Condition 1", "rotating_speed_rpm": 1800, "radial_load_n": 4000},
        "2": {"operating_condition": "Condition 2", "rotating_speed_rpm": 1650, "radial_load_n": 4200},
        "3": {"operating_condition": "Condition 3", "rotating_speed_rpm": 1500, "radial_load_n": 5000},
    }
    test_terminal_rul_seconds = {
        "Bearing1_3": 5730.0,
        "Bearing1_4": 339.0,
        "Bearing1_5": 1610.0,
        "Bearing1_6": 1460.0,
        "Bearing1_7": 7570.0,
        "Bearing2_3": 7530.0,
        "Bearing2_4": 1390.0,
        "Bearing2_5": 3090.0,
        "Bearing2_6": 1290.0,
        "Bearing2_7": 580.0,
        "Bearing3_3": 820.0,
    }

    dataset_name = "PHM2012"
    resource = DatasetResource(
        name="phm2012",
        homepage="https://data.nasa.gov/dataset/FEMTO-Bearing-Dataset/jujd-xjyk",
        download_url="https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip",
        description="IEEE PHM 2012 challenge bearing dataset hosted by NASA data portal.",
        notes="可以直接下载 zip，也可以从 NASA data portal 页面查看说明与镜像信息。",
    )

    def _build_entity_metadata(self, entity_path: Path) -> dict[str, object]:
        metadata = super()._build_entity_metadata(entity_path)
        split_name = self._infer_split_name(entity_path)
        metadata["split_name"] = split_name
        metadata["acceleration_sampling_points"] = 2560
        metadata["temperature_sample_rate_hz"] = 10.0

        condition_key = self._infer_condition_key(entity_path.name)
        if condition_key in self.condition_metadata:
            metadata.update(self.condition_metadata[condition_key])

        terminal_rul_seconds = self._terminal_rul_seconds(entity_path)
        if terminal_rul_seconds is not None:
            metadata["known_terminal_rul_seconds"] = terminal_rul_seconds
        return metadata

    def _iter_signal_files(self, entity_path: Path) -> Iterable[Path]:
        """
        return only acceleration snapshots for the primary sample timeline

        Parameters
        ----------
        entity_path : Path
            bearing directory

        Returns
        -------
        Iterable[Path]
            acceleration signal files
        """

        return [file_path for file_path in super()._iter_signal_files(entity_path) if file_path.name.startswith("acc_")]

    def _load_entity_frame(self, entity_path: Path) -> pd.DataFrame:
        """
        load acceleration snapshots and align temperature files by snapshot id

        Parameters
        ----------
        entity_path : Path
            bearing directory

        Returns
        -------
        pd.DataFrame
            aligned snapshot frame
        """

        records: list[dict[str, object]] = []
        temperature_files = {
            self._signal_sort_key(file_path)[0]: file_path
            for file_path in entity_path.rglob("temp_*.csv")
            if file_path.is_file()
        }

        for sample_index, file_path in enumerate(self._iter_signal_files(entity_path)):
            signal_frame = self._read_signal_file(file_path)
            horizontal_signal, vertical_signal = self._extract_channels(signal_frame)

            snapshot_id = self._signal_sort_key(file_path)[0]
            temperature_file = temperature_files.get(snapshot_id)
            temperature_signal = np.asarray([], dtype=float)
            if temperature_file is not None:
                temperature_frame = self._read_signal_file(temperature_file)
                temperature_signal = temperature_frame.iloc[:, -1].to_numpy(dtype=float)

            records.append(
                {
                    "sample_index": sample_index,
                    "timestamp": self._build_sample_timestamp(sample_index, file_path, signal_frame),
                    "rul": 0.0,
                    "Horizontal Vibration": horizontal_signal,
                    "Vertical Vibration": vertical_signal,
                    "Temperature": temperature_signal,
                    "source_file": file_path.name,
                    "temperature_file": temperature_file.name if temperature_file is not None else None,
                }
            )

        sample_frame = pd.DataFrame.from_records(records)
        if sample_frame.empty:
            raise ValueError(f"no acceleration files were found under {entity_path}")
        sample_frame = self._finalize_sample_frame(sample_frame, entity_path)
        terminal_rul_seconds = self._terminal_rul_seconds(entity_path)
        if terminal_rul_seconds is not None:
            sample_frame["rul"] = sample_frame["rul"] + terminal_rul_seconds
        return sample_frame

    def _infer_sample_rate(self, entity_path: Path) -> float:
        del entity_path
        return 25600.0

    def _extract_channels(self, signal_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if signal_frame.shape[1] >= 6:
            return signal_frame.iloc[:, 4].to_numpy(dtype=float), signal_frame.iloc[:, 5].to_numpy(dtype=float)
        return super()._extract_channels(signal_frame)

    def _build_sample_timestamp(
        self,
        sample_index: int,
        file_path: Path,
        signal_frame: pd.DataFrame,
    ) -> float:
        """
        build timestamp from PHM2012 time columns when present

        Parameters
        ----------
        sample_index : int
            chronological sample index
        file_path : Path
            source file path
        signal_frame : pd.DataFrame
            parsed acceleration frame

        Returns
        -------
        float
            timestamp in seconds
        """

        del file_path
        if signal_frame.shape[1] >= 4 and not signal_frame.empty:
            first_row = signal_frame.iloc[0, :4].to_numpy(dtype=float)
            if np.all(np.isfinite(first_row)):
                hour_value, minute_value, second_value, subsecond_value = first_row
                timestamp = (
                    (hour_value * 3600.0)
                    + (minute_value * 60.0)
                    + second_value
                    + self._parse_subsecond_value(subsecond_value)
                )
                return round(float(timestamp), 6)
        return float(sample_index) * self._sample_period_seconds(Path())

    def _sample_period_seconds(self, entity_path: Path) -> float:
        del entity_path
        return 10.0

    def _snapshot_duration_seconds(self, entity_path: Path) -> float:
        del entity_path
        return 0.1

    def _infer_split_name(self, entity_path: Path) -> str:
        for path in [entity_path.parent, *entity_path.parents]:
            if path.name in self.known_split_names:
                return path.name
        return entity_path.parent.name

    @classmethod
    def _infer_condition_key(cls, entity_id: str) -> str | None:
        match = re.match(r"Bearing(?P<condition>\d+)_\d+", entity_id, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group("condition")

    def _terminal_rul_seconds(self, entity_path: Path) -> float | None:
        if self._infer_split_name(entity_path) != "Test_set":
            return None
        return self.test_terminal_rul_seconds.get(entity_path.name)

    @staticmethod
    def _parse_subsecond_value(subsecond_value: float) -> float:
        if abs(subsecond_value) >= 1.0:
            return subsecond_value / 1_000_000.0
        return subsecond_value

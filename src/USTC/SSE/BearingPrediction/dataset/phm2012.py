"""
PHM2012 dataset loader module

this file is for loading IEEE PHM 2012 bearing dataset entities

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.dataset.base import BaseBearingLoader, DatasetResource


class PHM2012Loader(BaseBearingLoader):
    """
    Loader for the IEEE PHM 2012 FEMTO bearing dataset.
    """

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
        metadata["split_name"] = entity_path.parent.parent.name if entity_path.parent.parent else entity_path.parent.name
        metadata["operating_condition"] = entity_path.parent.name
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
                    "timestamp": sample_index,
                    "rul": max(len(records), 0),
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
        sample_frame["rul"] = np.arange(sample_frame.shape[0] - 1, -1, -1)
        return sample_frame

    def _infer_sample_rate(self, entity_path: Path) -> float:
        return 25600.0

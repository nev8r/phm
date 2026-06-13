"""
Dataset loader base module

this file is for defining base bearing dataset loader behavior

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.data.entities import BearingEntity


@dataclass(frozen=True)
class DatasetResource:
    """
    External dataset descriptor.

    Parameters
    ----------
    name : str
        dataset name
    homepage : str
        homepage url
    download_url : str | None
        archive download url
    description : str
        dataset description
    notes : str
        extra instructions
    """

    name: str
    homepage: str
    download_url: str | None
    description: str
    notes: str


class BaseBearingLoader:
    """
    Base class for XJTU-SY and PHM2012 dataset loaders.
    """

    dataset_name = "BaseDataset"
    resource = DatasetResource(
        name="base",
        homepage="",
        download_url=None,
        description="",
        notes="",
    )

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

    def list_entities(self) -> list[str]:
        """
        list entity identifiers found under the dataset root

        Returns
        -------
        list[str]
            entity identifiers
        """

        entity_paths = [path for path in self.data_root.rglob("*") if path.is_dir() and self._is_entity_path(path)]
        return sorted({path.name for path in entity_paths})

    def load_entity(self, entity_id: str, *, max_samples: int | None = None) -> BearingEntity:
        """
        load one bearing entity

        Parameters
        ----------
        entity_id : str
            entity id
        max_samples : int | None
            optional number of signal files to sample before reading

        Returns
        -------
        BearingEntity
            parsed entity
        """

        entity_path = self._resolve_entity_path(entity_id)
        sample_frame = self._load_entity_frame(entity_path, max_samples=max_samples)
        metadata = self._build_entity_metadata(entity_path)
        if "source_sample_count" in sample_frame.attrs:
            metadata["source_sample_count"] = int(sample_frame.attrs["source_sample_count"])
            metadata["used_sample_count"] = int(sample_frame.attrs["used_sample_count"])
        return BearingEntity(
            entity_id=entity_id,
            dataset_name=self.dataset_name,
            samples=sample_frame,
            sample_rate=self._infer_sample_rate(entity_path),
            metadata=metadata,
        )

    @classmethod
    def dataset_resource(cls) -> DatasetResource:
        """
        return dataset download descriptor

        Returns
        -------
        DatasetResource
            dataset descriptor
        """

        return cls.resource

    @classmethod
    def download(cls, output_dir: str | Path, *, extract: bool = True) -> Path:
        """
        download dataset archive when a direct url is available

        Parameters
        ----------
        output_dir : str | Path
            target directory
        extract : bool
            whether to extract zip archive

        Returns
        -------
        Path
            downloaded archive or extracted directory
        """

        resource = cls.dataset_resource()
        if resource.download_url is None:
            raise RuntimeError(f"{resource.name} does not provide a reliable direct download url. {resource.notes}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_name = resource.download_url.split("/")[-1].split("?")[0] or f"{resource.name}.zip"
        archive_path = output_dir / archive_name
        if not archive_path.exists():
            urlretrieve(resource.download_url, archive_path)
        if extract and archive_path.suffix.lower() == ".zip":
            extract_dir = output_dir / archive_path.stem
            if not extract_dir.exists():
                with zipfile.ZipFile(archive_path, "r") as archive_file:
                    archive_file.extractall(extract_dir)
            return extract_dir
        return archive_path

    def _resolve_entity_path(self, entity_id: str) -> Path:
        candidate_paths = [path for path in self.data_root.rglob(entity_id) if path.is_dir()]
        if not candidate_paths:
            raise FileNotFoundError(f"{entity_id} was not found under {self.data_root}")
        return sorted(candidate_paths)[0]

    def _load_entity_frame(self, entity_path: Path, *, max_samples: int | None = None) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        signal_files = list(self._iter_signal_files(entity_path))
        selected_files = self._select_signal_files(signal_files, max_samples)
        for sample_index, file_path in selected_files:
            signal_frame = self._read_signal_file(file_path)
            horizontal_signal, vertical_signal = self._extract_channels(signal_frame)
            records.append(
                {
                    "sample_index": sample_index,
                    "timestamp": self._build_sample_timestamp(sample_index, file_path, signal_frame),
                    "rul": 0.0,
                    "Horizontal Vibration": horizontal_signal,
                    "Vertical Vibration": vertical_signal,
                    "source_file": file_path.name,
                }
            )
        sample_frame = pd.DataFrame.from_records(records)
        if sample_frame.empty:
            raise ValueError(f"no signal files were found under {entity_path}")
        sample_frame.attrs["source_sample_count"] = len(signal_files)
        sample_frame.attrs["used_sample_count"] = len(selected_files)
        return self._finalize_sample_frame(sample_frame, entity_path)

    def _iter_signal_files(self, entity_path: Path) -> Iterable[Path]:
        candidate_files = [
            file_path
            for file_path in entity_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".txt"}
        ]
        return sorted(candidate_files, key=self._signal_sort_key)

    def _signal_sort_key(self, file_path: Path) -> tuple[int, str]:
        digits = re.findall(r"\d+", file_path.stem)
        numeric_order = int(digits[-1]) if digits else 0
        return numeric_order, file_path.name

    def _select_signal_files(self, signal_files: list[Path], max_samples: int | None) -> list[tuple[int, Path]]:
        """
        select signal files before reading large run-to-failure entities

        Parameters
        ----------
        signal_files : list[Path]
            sorted signal files
        max_samples : int | None
            optional maximum sample count

        Returns
        -------
        list[tuple[int, Path]]
            original chronological index and file path
        """

        indexed_files = list(enumerate(signal_files))
        if max_samples is None or max_samples <= 0 or len(indexed_files) <= max_samples:
            return indexed_files
        sample_indices = np.linspace(0, len(indexed_files) - 1, max_samples, dtype=int)
        return [indexed_files[index] for index in np.unique(sample_indices)]

    def _read_signal_file(self, file_path: Path) -> pd.DataFrame:
        for separator in [",", ";", r"\s+"]:
            try:
                signal_frame = pd.read_csv(file_path, header=None, sep=separator, engine="python")
                if signal_frame.shape[1] >= 2:
                    numeric_frame = signal_frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
                    if numeric_frame.shape[0] > 0:
                        return numeric_frame.reset_index(drop=True)
            except Exception:
                continue
        raise ValueError(f"failed to parse signal file: {file_path}")

    def _extract_channels(self, signal_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if signal_frame.shape[1] == 2:
            return signal_frame.iloc[:, 0].to_numpy(dtype=float), signal_frame.iloc[:, 1].to_numpy(dtype=float)
        return signal_frame.iloc[:, -2].to_numpy(dtype=float), signal_frame.iloc[:, -1].to_numpy(dtype=float)

    def _infer_sample_rate(self, entity_path: Path) -> float:
        return 25600.0

    def _build_entity_metadata(self, entity_path: Path) -> dict[str, object]:
        return {
            "entity_path": str(entity_path),
            "sample_rate_hz": self._infer_sample_rate(entity_path),
            "sampling_period_seconds": self._sample_period_seconds(entity_path),
            "snapshot_duration_seconds": self._snapshot_duration_seconds(entity_path),
            "rul_unit": self._rul_unit(entity_path),
        }

    def _is_entity_path(self, path: Path) -> bool:
        return path.name.lower().startswith("bearing")

    def _build_sample_timestamp(
        self,
        sample_index: int,
        file_path: Path,
        signal_frame: pd.DataFrame,
    ) -> float:
        """
        build sample timestamp for one signal snapshot

        Parameters
        ----------
        sample_index : int
            chronological sample index
        file_path : Path
            source file path
        signal_frame : pd.DataFrame
            parsed signal frame

        Returns
        -------
        float
            timestamp in seconds
        """

        del signal_frame
        return float(sample_index) * self._sample_period_seconds(file_path.parent)

    def _finalize_sample_frame(self, sample_frame: pd.DataFrame, entity_path: Path) -> pd.DataFrame:
        """
        assign elapsed time and RUL columns after all snapshots are parsed

        Parameters
        ----------
        sample_frame : pd.DataFrame
            raw snapshot records
        entity_path : Path
            bearing directory

        Returns
        -------
        pd.DataFrame
            finalized snapshot records
        """

        sample_period_seconds = self._sample_period_seconds(entity_path)
        timestamps = sample_frame["timestamp"].astype(float).to_numpy()
        elapsed_seconds = timestamps - timestamps[0]
        if np.any(np.diff(elapsed_seconds) < 0):
            elapsed_seconds = np.arange(sample_frame.shape[0], dtype=float) * sample_period_seconds

        sample_frame["timestamp"] = np.round(timestamps, 6)
        sample_frame["elapsed_seconds"] = np.round(elapsed_seconds, 6)
        del sample_period_seconds
        sample_frame["rul"] = np.round(float(np.max(elapsed_seconds)) - elapsed_seconds, 6)
        return sample_frame

    def _sample_period_seconds(self, entity_path: Path) -> float:
        """
        infer interval between two consecutive signal snapshots

        Parameters
        ----------
        entity_path : Path
            bearing directory

        Returns
        -------
        float
            sampling period in seconds
        """

        del entity_path
        return 1.0

    def _snapshot_duration_seconds(self, entity_path: Path) -> float:
        """
        infer duration covered by one signal snapshot

        Parameters
        ----------
        entity_path : Path
            bearing directory

        Returns
        -------
        float
            snapshot duration in seconds
        """

        del entity_path
        return 1.0

    def _rul_unit(self, entity_path: Path) -> str:
        """
        return unit used by RUL labels

        Parameters
        ----------
        entity_path : Path
            bearing directory

        Returns
        -------
        str
            RUL unit name
        """

        del entity_path
        return "seconds"

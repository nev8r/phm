"""
Dataset loader base module

this file is for defining base bearing dataset loader behavior

created by cyy

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

    def load_entity(self, entity_id: str) -> BearingEntity:
        """
        load one bearing entity

        Parameters
        ----------
        entity_id : str
            entity id

        Returns
        -------
        BearingEntity
            parsed entity
        """

        entity_path = self._resolve_entity_path(entity_id)
        sample_frame = self._load_entity_frame(entity_path)
        return BearingEntity(
            entity_id=entity_id,
            dataset_name=self.dataset_name,
            samples=sample_frame,
            sample_rate=self._infer_sample_rate(entity_path),
            metadata=self._build_entity_metadata(entity_path),
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

    def _load_entity_frame(self, entity_path: Path) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for sample_index, file_path in enumerate(self._iter_signal_files(entity_path)):
            signal_frame = self._read_signal_file(file_path)
            horizontal_signal, vertical_signal = self._extract_channels(signal_frame)
            records.append(
                {
                    "sample_index": sample_index,
                    "timestamp": sample_index,
                    "rul": max(len(records), 0),
                    "Horizontal Vibration": horizontal_signal,
                    "Vertical Vibration": vertical_signal,
                    "source_file": file_path.name,
                }
            )
        sample_frame = pd.DataFrame.from_records(records)
        if sample_frame.empty:
            raise ValueError(f"no signal files were found under {entity_path}")
        sample_frame["rul"] = np.arange(sample_frame.shape[0] - 1, -1, -1)
        return sample_frame

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
        return {"entity_path": str(entity_path)}

    def _is_entity_path(self, path: Path) -> bool:
        return path.name.lower().startswith("bearing")

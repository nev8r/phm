"""
Sample index builder.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.index.SampleIndex import SAMPLE_INDEX_COLUMNS
from USTC.SSE.BearingPrediction.infra.metadata.BearingMeta import BearingMeta
from USTC.SSE.BearingPrediction.infra.metadata.PHM2012Meta import PHM2012Meta
from USTC.SSE.BearingPrediction.infra.metadata.XJTUSYMeta import XJTUSYMeta


class IndexBuilder:
    """
    Build sample-level indices for supported bearing datasets.
    """

    def build(self, cfg: DictConfig) -> pd.DataFrame:
        dataset_name = str(OmegaConf.select(cfg, "dataset.name", default=""))
        root = Path(str(OmegaConf.select(cfg, "dataset.root", default=""))).expanduser()
        if not root:
            raise ValueError("dataset.root is required")

        if dataset_name == XJTUSYMeta.DATASET_NAME:
            return self._build_xjtu_sy(root)
        if dataset_name == PHM2012Meta.DATASET_NAME:
            return self._build_phm2012(root)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _build_xjtu_sy(self, root: Path) -> pd.DataFrame:
        dataset_meta = XJTUSYMeta()
        rows: List[Dict] = []
        for condition_id in dataset_meta.CONDITIONS:
            condition_dir = root / condition_id
            if not condition_dir.is_dir():
                continue
            for bearing_dir in _iter_dirs(condition_dir):
                meta = dataset_meta.get_bearing_meta(bearing_dir.name)
                csv_files = _sort_by_first_int(bearing_dir.glob("*.csv"))
                rows.extend(
                    self._rows_for_files(
                        meta=meta,
                        files=csv_files,
                        source_group=None,
                    )
                )
        return _to_index_frame(rows)

    def _build_phm2012(self, root: Path) -> pd.DataFrame:
        dataset_meta = PHM2012Meta()
        rows: List[Dict] = []
        for source_group in ("Learning_set", "Full_Test_Set"):
            source_dir = root / source_group
            if not source_dir.is_dir():
                continue
            for bearing_dir in _iter_dirs(source_dir):
                meta = dataset_meta.get_bearing_meta(bearing_dir.name)
                acc_files = [path for path in bearing_dir.iterdir() if path.is_file() and path.name.startswith("acc")]
                rows.extend(
                    self._rows_for_files(
                        meta=meta,
                        files=_sort_by_first_int(acc_files),
                        source_group=source_group,
                    )
                )
        return _to_index_frame(rows)

    def _rows_for_files(
            self,
            meta: BearingMeta,
            files: Iterable[Path],
            source_group: Optional[str],
    ) -> List[Dict]:
        rows = []
        for timestep, path in enumerate(files):
            sample_id = _format_sample_id(path.name)
            rows.append({
                "sample_uid": f"{meta.dataset}::{meta.bearing_id}::{sample_id}",
                "dataset": meta.dataset,
                "bearing_id": meta.bearing_id,
                "condition_id": meta.condition_id,
                "source_group": source_group,
                "sample_id": sample_id,
                "timestep": timestep,
                "file_path": str(path),
                "sampling_rate": meta.sampling_rate,
                "expected_points": meta.expected_points_per_sample,
                "sample_interval_seconds": meta.sample_interval_seconds,
                "channel_names": ",".join(meta.channels),
                "speed_hz": meta.speed_hz,
                "load_n": meta.load_n,
                "fault_element": _format_fault_element(meta.fault_element),
                "is_run_to_failure": True,
            })
        return rows


def extract_first_int(path_or_name: str) -> int:
    match = re.search(r"\d+", str(path_or_name))
    return int(match.group()) if match else 0


def _format_sample_id(file_name: str) -> str:
    return f"{extract_first_int(file_name):06d}"


def _format_fault_element(fault_element) -> Optional[str]:
    if fault_element is None:
        return None
    return ",".join(fault_element)


def _iter_dirs(root: Path) -> List[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: _bearing_sort_key(path.name))


def _sort_by_first_int(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=lambda path: extract_first_int(path.name))


def _bearing_sort_key(name: str):
    numbers = [int(value) for value in re.findall(r"\d+", name)]
    return numbers or [0]


def _to_index_frame(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SAMPLE_INDEX_COLUMNS)

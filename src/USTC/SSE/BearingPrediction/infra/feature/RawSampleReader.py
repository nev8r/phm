"""
Raw sample reader.

This module reads one sample file referenced by the sample index and returns a
time-by-channel vibration array.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


class RawSampleReader:
    CHANNELS = ["h", "v"]

    def read(self, sample_row) -> Tuple[np.ndarray, List[str]]:
        dataset = _row_value(sample_row, "dataset")
        file_path = Path(str(_row_value(sample_row, "file_path")))

        if dataset == "XJTU-SY":
            signal = self._read_xjtu(file_path)
        elif dataset == "PHM2012":
            signal = self._read_phm2012(file_path)
        else:
            raise ValueError(f"Unsupported dataset for raw sample reading: {dataset}")

        if signal.ndim != 2 or signal.shape[1] != 2:
            raise ValueError(f"Expected two-channel signal, got shape={signal.shape}")
        if not np.isfinite(signal).all():
            raise ValueError(f"Raw sample contains NaN or Inf: {file_path}")
        return signal, list(self.CHANNELS)

    def _read_xjtu(self, file_path: Path) -> np.ndarray:
        df = pd.read_csv(file_path)
        expected = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
        if all(column in df.columns for column in expected):
            data = df[expected]
        else:
            data = df.iloc[:, :2]
        return data.to_numpy(dtype=float)

    def _read_phm2012(self, file_path: Path) -> np.ndarray:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            df = pd.read_csv(file_path, header=None, sep=";")
        if df.shape[1] < 2:
            raise ValueError(f"PHM2012 acc file must contain at least two columns: {file_path}")
        return df.iloc[:, -2:].to_numpy(dtype=float)


def _row_value(sample_row, key: str):
    if isinstance(sample_row, dict):
        return sample_row[key]
    return sample_row[key]

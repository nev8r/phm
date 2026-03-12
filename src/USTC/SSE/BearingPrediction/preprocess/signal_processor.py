"""
Signal preprocessor module

this file is for preprocessing vibration signal data

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.config import DataConfig


class BearingSignalPreprocessor:
    """
    Clean, normalize, and window vibration signals.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def build_windowed_samples(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        preprocess raw signals and generate sliding window samples

        Parameters
        ----------
        dataset : pd.DataFrame
            raw cycle level dataset

        Returns
        -------
        pd.DataFrame
            window level dataset
        """

        records: list[dict[str, object]] = []

        for row in dataset.itertuples(index=False):
            cleaned_signal = self._clip_outliers(np.asarray(row.signal, dtype=float))
            normalized_signal = self._normalize_signal(cleaned_signal)

            for window_index, signal_window in enumerate(self._window_signal(normalized_signal)):
                records.append(
                    {
                        "bearing_id": row.bearing_id,
                        "cycle": int(row.cycle),
                        "window_index": window_index,
                        "temperature": float(row.temperature),
                        "load": float(row.load),
                        "health_index": float(row.health_index),
                        "failure_cycle": int(row.failure_cycle),
                        "rul": int(row.rul),
                        "duration": int(row.duration),
                        "event": int(row.event),
                        "signal_window": signal_window,
                    }
                )

        windowed_samples = pd.DataFrame.from_records(records)
        return windowed_samples

    @staticmethod
    def _clip_outliers(signal_values: np.ndarray, clip_sigma: float = 3.5) -> np.ndarray:
        """
        clip signal outliers using robust median absolute deviation

        Parameters
        ----------
        signal_values : np.ndarray
            input signal
        clip_sigma : float
            clipping factor

        Returns
        -------
        np.ndarray
            cleaned signal
        """

        median_value = float(np.median(signal_values))
        mad_value = float(np.median(np.abs(signal_values - median_value)))
        if mad_value == 0.0:
            return signal_values

        scale = 1.4826 * mad_value
        lower_bound = median_value - (clip_sigma * scale)
        upper_bound = median_value + (clip_sigma * scale)
        return np.clip(signal_values, lower_bound, upper_bound)

    @staticmethod
    def _normalize_signal(signal_values: np.ndarray) -> np.ndarray:
        """
        normalize signal with z-score scaling

        Parameters
        ----------
        signal_values : np.ndarray
            input signal

        Returns
        -------
        np.ndarray
            normalized signal
        """

        mean_value = float(np.mean(signal_values))
        standard_deviation = float(np.std(signal_values))
        if standard_deviation == 0.0:
            return signal_values - mean_value
        return (signal_values - mean_value) / standard_deviation

    def _window_signal(self, signal_values: np.ndarray) -> list[np.ndarray]:
        """
        create overlapping sliding windows from signal data

        Parameters
        ----------
        signal_values : np.ndarray
            normalized signal

        Returns
        -------
        list[np.ndarray]
            sliding windows
        """

        windows: list[np.ndarray] = []
        last_start = max(signal_values.size - self.data_config.window_size, 0)

        for start_index in range(0, last_start + 1, self.data_config.window_stride):
            end_index = start_index + self.data_config.window_size
            windows.append(signal_values[start_index:end_index])

        if not windows:
            windows.append(signal_values)
        return windows


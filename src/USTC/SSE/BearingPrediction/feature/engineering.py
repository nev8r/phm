"""
Feature engineering module

this file is for extracting configurable bearing signal features

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """
    Feature extraction configuration.

    Parameters
    ----------
    sample_rate : float
        signal sample rate
    enabled_features : tuple[str, ...]
        selected feature names
    """

    sample_rate: float
    enabled_features: tuple[str, ...] = (
        "mean",
        "variance",
        "standard_deviation",
        "rms",
        "peak",
        "peak_to_peak",
        "absolute_mean",
        "shape_factor",
        "crest_factor",
        "impulse_factor",
        "margin_factor",
        "clearance_factor",
        "kurtosis",
        "skewness",
        "dominant_frequency",
        "spectrum_energy",
        "spectral_centroid",
        "spectral_rms_frequency",
        "spectral_entropy",
    )


class SignalFeatureExtractor:
    """
    Extract time and frequency domain features from a set of windows.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def extract(self, windows: list[np.ndarray]) -> pd.DataFrame:
        """
        extract a dataframe of features

        Parameters
        ----------
        windows : list[np.ndarray]
            signal windows

        Returns
        -------
        pd.DataFrame
            feature dataframe
        """

        records = [self.extract_one(window_values) for window_values in windows]
        return pd.DataFrame.from_records(records)

    def extract_one(self, signal_values: np.ndarray) -> dict[str, float]:
        """
        extract one feature record

        Parameters
        ----------
        signal_values : np.ndarray
            signal window

        Returns
        -------
        dict[str, float]
            feature values
        """

        absolute_values = np.abs(signal_values)
        absolute_mean = float(np.mean(absolute_values))
        rms_value = float(np.sqrt(np.mean(np.square(signal_values))))
        peak_value = float(np.max(absolute_values))
        square_root_amplitude_mean = float(np.mean(np.sqrt(absolute_values)))
        feature_values = {
            "mean": float(np.mean(signal_values)),
            "variance": float(np.var(signal_values)),
            "standard_deviation": float(np.std(signal_values)),
            "rms": rms_value,
            "peak": peak_value,
            "peak_to_peak": float(np.ptp(signal_values)),
            "absolute_mean": absolute_mean,
            "kurtosis": self._kurtosis(signal_values),
            "skewness": self._skewness(signal_values),
        }
        feature_values["shape_factor"] = self._safe_divide(rms_value, absolute_mean)
        feature_values["crest_factor"] = self._safe_divide(peak_value, rms_value)
        feature_values["impulse_factor"] = self._safe_divide(peak_value, absolute_mean)
        feature_values["margin_factor"] = self._safe_divide(peak_value, square_root_amplitude_mean**2)
        feature_values["clearance_factor"] = feature_values["margin_factor"]
        feature_values.update(self._frequency_features(signal_values))
        return {name: feature_values[name] for name in self.config.enabled_features if name in feature_values}

    def build_health_indicator(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """
        build a composite health indicator from selected features

        Parameters
        ----------
        feature_frame : pd.DataFrame
            feature dataframe

        Returns
        -------
        np.ndarray
            normalized health indicator
        """

        selected_columns = [column_name for column_name in ["rms", "kurtosis", "crest_factor", "spectrum_energy"] if column_name in feature_frame.columns]
        if not selected_columns:
            values = feature_frame.iloc[:, 0].to_numpy(dtype=float)
        else:
            zscore_frame = (feature_frame[selected_columns] - feature_frame[selected_columns].mean()) / (feature_frame[selected_columns].std() + 1e-8)
            values = zscore_frame.mean(axis=1).to_numpy(dtype=float)
        values = np.asarray(values, dtype=float)
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
        return normalized_values

    def _frequency_features(self, signal_values: np.ndarray) -> dict[str, float]:
        spectrum = np.fft.rfft(signal_values)
        magnitudes = np.abs(spectrum)
        power = np.square(magnitudes)
        power_sum = float(power.sum())
        frequencies = np.fft.rfftfreq(signal_values.size, d=1.0 / self.config.sample_rate)
        dominant_index = int(np.argmax(magnitudes[1:]) + 1) if magnitudes.size > 1 else 0
        normalized_power = power / (power_sum + 1e-8)
        spectral_centroid = self._safe_divide(float(np.sum(frequencies * power)), power_sum)
        spectral_rms_frequency = float(np.sqrt(self._safe_divide(float(np.sum(np.square(frequencies) * power)), power_sum)))
        spectral_entropy = float(-np.sum(normalized_power * np.log2(normalized_power + 1e-8)))
        return {
            "dominant_frequency": float(frequencies[dominant_index]) if frequencies.size else 0.0,
            "spectrum_energy": power_sum,
            "spectral_centroid": spectral_centroid,
            "spectral_rms_frequency": spectral_rms_frequency,
            "spectral_entropy": spectral_entropy,
        }

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-8:
            return 0.0
        return float(numerator / denominator)

    def _kurtosis(self, signal_values: np.ndarray) -> float:
        centered_values = signal_values - np.mean(signal_values)
        variance_value = np.var(centered_values)
        if variance_value == 0.0:
            return 0.0
        return float(np.mean(np.power(centered_values, 4)) / math.pow(variance_value, 2))

    def _skewness(self, signal_values: np.ndarray) -> float:
        centered_values = signal_values - np.mean(signal_values)
        std_value = np.std(centered_values)
        if std_value == 0.0:
            return 0.0
        return float(np.mean(np.power(centered_values / std_value, 3)))

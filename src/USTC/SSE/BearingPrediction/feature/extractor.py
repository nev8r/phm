"""
Feature extractor module

this file is for extracting time and frequency domain features

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.config import DataConfig

try:
    from tsfresh.feature_extraction import feature_calculators
except ImportError:  # pragma: no cover - handled at runtime when dependency is unavailable
    feature_calculators = None


class BearingFeatureExtractor:
    """
    Extract time-domain and frequency-domain features from vibration windows.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def extract_features(self, windowed_samples: pd.DataFrame) -> pd.DataFrame:
        """
        extract feature table from sliding window samples

        Parameters
        ----------
        windowed_samples : pd.DataFrame
            preprocessed window level dataset

        Returns
        -------
        pd.DataFrame
            feature table for model training
        """

        records: list[dict[str, float | int | str]] = []

        for row in windowed_samples.itertuples(index=False):
            signal_window = np.asarray(row.signal_window, dtype=float)
            time_features = self._extract_time_features(signal_window)
            frequency_features = self._extract_frequency_features(signal_window)
            tsfresh_features = self._extract_tsfresh_features(signal_window)
            records.append(
                {
                    "bearing_id": row.bearing_id,
                    "cycle": int(row.cycle),
                    "window_index": int(row.window_index),
                    "temperature": float(row.temperature),
                    "load": float(row.load),
                    "health_index": float(row.health_index),
                    "failure_cycle": int(row.failure_cycle),
                    "rul": int(row.rul),
                    "duration": int(row.duration),
                    "event": int(row.event),
                    **time_features,
                    **frequency_features,
                    **tsfresh_features,
                }
            )

        feature_table = pd.DataFrame.from_records(records)
        return feature_table

    def _extract_time_features(self, signal_values: np.ndarray) -> dict[str, float]:
        """
        extract time domain features

        Parameters
        ----------
        signal_values : ndarray
            vibration signal

        Returns
        -------
        dict
            extracted features
        """

        mean_value = float(np.mean(signal_values))
        variance_value = float(np.var(signal_values))
        rms_value = float(np.sqrt(np.mean(np.square(signal_values))))
        peak_value = float(np.max(np.abs(signal_values)))
        peak_to_peak_value = float(np.ptp(signal_values))
        kurtosis_value = self._calculate_kurtosis(signal_values)
        skewness_value = self._calculate_skewness(signal_values)
        crest_factor = peak_value / rms_value if rms_value else 0.0

        return {
            "mean": mean_value,
            "variance": variance_value,
            "rms": rms_value,
            "peak": peak_value,
            "peak_to_peak": peak_to_peak_value,
            "kurtosis": kurtosis_value,
            "skewness": skewness_value,
            "crest_factor": float(crest_factor),
        }

    def _extract_frequency_features(self, signal_values: np.ndarray) -> dict[str, float]:
        """
        extract frequency domain features with fft

        Parameters
        ----------
        signal_values : np.ndarray
            vibration signal

        Returns
        -------
        dict[str, float]
            frequency domain features
        """

        spectrum = np.fft.rfft(signal_values)
        magnitudes = np.abs(spectrum)
        power_spectrum = np.square(magnitudes)
        frequency_axis = np.fft.rfftfreq(signal_values.size, d=1.0 / self.data_config.sample_rate)

        dominant_index = int(np.argmax(magnitudes[1:]) + 1) if magnitudes.size > 1 else 0
        total_power = float(np.sum(power_spectrum))
        normalized_power = power_spectrum / (np.sum(power_spectrum) + 1e-12)
        spectral_centroid = float(np.sum(frequency_axis * magnitudes) / (np.sum(magnitudes) + 1e-12))
        spectral_entropy = float(-np.sum(normalized_power * np.log2(normalized_power + 1e-12)))

        return {
            "dominant_frequency": float(frequency_axis[dominant_index]) if frequency_axis.size else 0.0,
            "spectrum_energy": total_power,
            "spectral_centroid": spectral_centroid,
            "spectral_entropy": spectral_entropy,
        }

    def _extract_tsfresh_features(self, signal_values: np.ndarray) -> dict[str, float]:
        """
        extract selected tsfresh style features

        Parameters
        ----------
        signal_values : np.ndarray
            vibration signal

        Returns
        -------
        dict[str, float]
            tsfresh feature values
        """

        if feature_calculators is None:
            absolute_energy = float(np.sum(np.square(signal_values)))
            lag_one_autocorrelation = float(np.corrcoef(signal_values[:-1], signal_values[1:])[0, 1]) if signal_values.size > 1 else 0.0
            return {
                "absolute_energy": absolute_energy,
                "autocorrelation_lag1": lag_one_autocorrelation,
            }

        absolute_energy = float(feature_calculators.abs_energy(signal_values))
        lag_one_autocorrelation = float(feature_calculators.autocorrelation(signal_values, lag=1))
        return {
            "absolute_energy": absolute_energy,
            "autocorrelation_lag1": lag_one_autocorrelation,
        }

    @staticmethod
    def _calculate_kurtosis(signal_values: np.ndarray) -> float:
        """
        calculate kurtosis without scipy dependency

        Parameters
        ----------
        signal_values : np.ndarray
            vibration signal

        Returns
        -------
        float
            kurtosis value
        """

        centered_values = signal_values - np.mean(signal_values)
        variance_value = np.var(centered_values)
        if variance_value == 0.0:
            return 0.0
        return float(np.mean(np.power(centered_values, 4)) / math.pow(variance_value, 2))

    @staticmethod
    def _calculate_skewness(signal_values: np.ndarray) -> float:
        """
        calculate skewness without scipy dependency

        Parameters
        ----------
        signal_values : np.ndarray
            vibration signal

        Returns
        -------
        float
            skewness value
        """

        centered_values = signal_values - np.mean(signal_values)
        standard_deviation = np.std(centered_values)
        if standard_deviation == 0.0:
            return 0.0
        return float(np.mean(np.power(centered_values / standard_deviation, 3)))


"""
Time domain feature processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

import numpy as np
from numpy import ndarray
from scipy.stats import kurtosis, skew

from USTC.SSE.BearingPrediction.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor


class TimeDomainFeatureProcessor(ABCBaseProcessor):
    """
    Extract a compact vector of bearing vibration time-domain statistics.
    """

    supported_features = {
        "mean",
        "std",
        "var",
        "rms",
        "mean_abs",
        "max",
        "min",
        "ptp",
        "skewness",
        "kurtosis",
        "crest_factor",
        "shape_factor",
        "impulse_factor",
        "clearance_factor",
    }

    def __init__(self, features=("mean", "std", "rms", "mean_abs", "ptp",
                                 "skewness", "kurtosis", "crest_factor")):
        self.features = tuple(features)
        self._validate()

    @property
    def name(self) -> str:
        return "Time_Domain_Feature"

    def run(self, source: ndarray) -> ndarray:
        signal, input_was_1d = FFTMagnitudeProcessor._as_time_channels(source)
        channel_features = [self._extract_channel(signal[:, i]) for i in range(signal.shape[1])]
        result = np.vstack(channel_features)
        return result[0] if input_was_1d else result

    def _extract_channel(self, signal: ndarray) -> ndarray:
        abs_signal = np.abs(signal)
        rms = np.sqrt(np.mean(np.square(signal)))
        mean_abs = np.mean(abs_signal)
        peak = np.max(abs_signal)
        sqrt_abs_mean = np.mean(np.sqrt(abs_signal))

        values = {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "var": np.var(signal),
            "rms": rms,
            "mean_abs": mean_abs,
            "max": np.max(signal),
            "min": np.min(signal),
            "ptp": np.ptp(signal),
            "skewness": skew(signal, bias=False),
            "kurtosis": kurtosis(signal, bias=False),
            "crest_factor": self._safe_divide(peak, rms),
            "shape_factor": self._safe_divide(rms, mean_abs),
            "impulse_factor": self._safe_divide(peak, mean_abs),
            "clearance_factor": self._safe_divide(peak, np.square(sqrt_abs_mean)),
        }
        result = np.array([values[name] for name in self.features], dtype=float)
        return np.nan_to_num(result)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def _validate(self) -> None:
        unknown = set(self.features) - self.supported_features
        if unknown:
            raise ValueError(f"Unsupported time-domain features: {sorted(unknown)}")

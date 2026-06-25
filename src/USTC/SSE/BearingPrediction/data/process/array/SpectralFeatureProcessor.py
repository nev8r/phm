"""
Spectral feature processor module

Purpose: load, label, or process bearing vibration data
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor


class SpectralFeatureProcessor(ABCBaseProcessor):
    """
    Extract compact frequency-domain statistics from vibration signals.
    """

    supported_features = {
        "centroid",
        "bandwidth",
        "rms_frequency",
        "peak_frequency",
        "entropy",
        "flatness",
        "rolloff",
    }

    def __init__(self, sampling_rate, features=("centroid", "bandwidth", "peak_frequency", "entropy"),
                 n_fft=None, include_dc=False, rolloff_ratio=0.85):
        self.sampling_rate = sampling_rate
        self.features = tuple(features)
        self.n_fft = n_fft
        self.include_dc = include_dc
        self.rolloff_ratio = rolloff_ratio
        self._validate()

    @property
    def name(self) -> str:
        return "Spectral_Feature"

    def run(self, source: ndarray) -> ndarray:
        signal, input_was_1d = FFTMagnitudeProcessor._as_time_channels(source)
        n_fft = self.n_fft or signal.shape[0]
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / self.sampling_rate)
        spectrum = np.fft.rfft(signal, n=n_fft, axis=0)
        power = np.square(np.abs(spectrum))

        if not self.include_dc:
            frequencies = frequencies[1:]
            power = power[1:]

        channel_features = [self._extract_channel(frequencies, power[:, i]) for i in range(power.shape[1])]
        result = np.vstack(channel_features)
        return result[0] if input_was_1d else result

    def _extract_channel(self, frequencies: ndarray, power: ndarray) -> ndarray:
        eps = np.finfo(float).eps
        total = np.sum(power)
        if total <= eps:
            return np.zeros(len(self.features), dtype=float)

        probability = power / total
        centroid = np.sum(frequencies * probability)
        bandwidth = np.sqrt(np.sum(np.square(frequencies - centroid) * probability))
        rms_frequency = np.sqrt(np.sum(np.square(frequencies) * probability))
        peak_frequency = frequencies[int(np.argmax(power))]
        entropy = self._entropy(probability)
        flatness = np.exp(np.mean(np.log(power + eps))) / (np.mean(power) + eps)
        rolloff = self._rolloff(frequencies, power, total)

        values = {
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rms_frequency": rms_frequency,
            "peak_frequency": peak_frequency,
            "entropy": entropy,
            "flatness": flatness,
            "rolloff": rolloff,
        }
        return np.array([values[name] for name in self.features], dtype=float)

    @staticmethod
    def _entropy(probability: ndarray) -> float:
        positive = probability[probability > 0]
        if positive.size <= 1:
            return 0.0
        return float(-np.sum(positive * np.log(positive)) / np.log(probability.size))

    def _rolloff(self, frequencies: ndarray, power: ndarray, total: float) -> float:
        threshold = self.rolloff_ratio * total
        index = int(np.searchsorted(np.cumsum(power), threshold, side="left"))
        index = min(index, frequencies.size - 1)
        return float(frequencies[index])

    def _validate(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0")
        unknown = set(self.features) - self.supported_features
        if unknown:
            raise ValueError(f"Unsupported spectral features: {sorted(unknown)}")
        if not 0 < self.rolloff_ratio <= 1:
            raise ValueError("rolloff_ratio must be in (0, 1]")

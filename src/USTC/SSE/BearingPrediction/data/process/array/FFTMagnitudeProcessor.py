"""
Fft magnitude processor module

Purpose: load, label, or process bearing vibration data
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.ABCBaseProcessor import ABCBaseProcessor


class FFTMagnitudeProcessor(ABCBaseProcessor):
    """
    Convert a vibration signal into a one-sided FFT magnitude spectrum.
    """

    def __init__(self, sampling_rate=None, n_fft=None, n_bins=None,
                 include_dc=False, log_scale=False, window=None, normalize=True):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.n_bins = n_bins
        self.include_dc = include_dc
        self.log_scale = log_scale
        self.window = window
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "FFT_Magnitude"

    def run(self, source: ndarray) -> ndarray:
        signal, input_was_1d = self._as_time_channels(source)
        n_fft = self.n_fft or signal.shape[0]
        if n_fft <= 0:
            raise ValueError("n_fft must be greater than 0")

        signal = self._apply_window(signal)
        spectrum = np.abs(np.fft.rfft(signal, n=n_fft, axis=0))
        if self.normalize:
            spectrum = spectrum / n_fft
        spectrum = self._select_bins(spectrum)
        if self.log_scale:
            spectrum = np.log1p(spectrum)

        return spectrum[:, 0] if input_was_1d else spectrum

    def frequency_bins(self, input_length: int) -> ndarray:
        n_fft = self.n_fft or input_length
        if n_fft <= 0:
            raise ValueError("n_fft must be greater than 0")
        sampling_rate = self.sampling_rate if self.sampling_rate is not None else 1.0
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
        return self._select_bins(frequencies)

    @staticmethod
    def _as_time_channels(source: ndarray) -> tuple[ndarray, bool]:
        signal = np.asarray(source, dtype=float)
        if signal.ndim == 1:
            return signal.reshape(-1, 1), True
        if signal.ndim == 2:
            return signal, False
        raise ValueError("source must be a 1-D signal or a 2-D array shaped as (time, channels)")

    def _apply_window(self, signal: ndarray) -> ndarray:
        if self.window is None:
            return signal
        window_name = self.window.lower()
        if window_name == "hann":
            weights = np.hanning(signal.shape[0])
        elif window_name == "hamming":
            weights = np.hamming(signal.shape[0])
        else:
            raise ValueError("window must be None, 'hann', or 'hamming'")
        return signal * weights.reshape(-1, 1)

    def _select_bins(self, values: ndarray) -> ndarray:
        selected = values if self.include_dc else values[1:]
        if self.n_bins is not None:
            if self.n_bins <= 0:
                raise ValueError("n_bins must be greater than 0")
            selected = selected[:self.n_bins]
        return selected

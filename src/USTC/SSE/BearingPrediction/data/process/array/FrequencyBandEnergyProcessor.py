"""
Frequency band energy processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.ABCBaseProcessor import ABCBaseProcessor
from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor


class FrequencyBandEnergyProcessor(ABCBaseProcessor):
    """
    Sum FFT power inside configured frequency bands.
    """

    def __init__(self, sampling_rate, bands, n_fft=None, relative=False, include_dc=True):
        self.sampling_rate = sampling_rate
        self.bands = tuple(bands)
        self.n_fft = n_fft
        self.relative = relative
        self.include_dc = include_dc
        self._validate()

    @property
    def name(self) -> str:
        return "Frequency_Band_Energy"

    def run(self, source: ndarray) -> ndarray:
        signal, input_was_1d = FFTMagnitudeProcessor._as_time_channels(source)
        n_fft = self.n_fft or signal.shape[0]
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / self.sampling_rate)
        spectrum = np.fft.rfft(signal, n=n_fft, axis=0)
        power = np.square(np.abs(spectrum))

        if not self.include_dc:
            frequencies = frequencies[1:]
            power = power[1:]

        values = []
        for low, high in self.bands:
            mask = (frequencies >= low) & (frequencies < high)
            values.append(np.sum(power[mask], axis=0))
        energy = np.vstack(values)

        if self.relative:
            total = np.sum(energy, axis=0, keepdims=True)
            energy = np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)

        if input_was_1d:
            return energy[:, 0]
        return energy.T

    def _validate(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0")
        if not self.bands:
            raise ValueError("bands must contain at least one frequency range")
        for low, high in self.bands:
            if low < 0 or high <= low:
                raise ValueError("each band must satisfy 0 <= low < high")

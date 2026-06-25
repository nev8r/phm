"""
array package initialization module

Purpose: load, label, or process bearing vibration data
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor
from USTC.SSE.BearingPrediction.data.process.array.FrequencyBandEnergyProcessor import FrequencyBandEnergyProcessor
from USTC.SSE.BearingPrediction.data.process.array.SpectralFeatureProcessor import SpectralFeatureProcessor
from USTC.SSE.BearingPrediction.data.process.array.TimeDomainFeatureProcessor import TimeDomainFeatureProcessor

__all__ = [
    "FFTMagnitudeProcessor",
    "FrequencyBandEnergyProcessor",
    "SpectralFeatureProcessor",
    "TimeDomainFeatureProcessor",
]

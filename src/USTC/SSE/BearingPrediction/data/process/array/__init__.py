"""
array package initialization module

this file is for exposing array package interfaces

created by cyj

copyright USTC

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

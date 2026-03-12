"""
Feature package

this file is for exposing feature engineering components

created by cyy

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.feature.extractor import BearingFeatureExtractor
from USTC.SSE.BearingPrediction.feature.engineering import FeatureConfig, SignalFeatureExtractor

__all__ = ["BearingFeatureExtractor", "FeatureConfig", "SignalFeatureExtractor"]


"""
Feature package

this file is for exposing feature engineering components

created by cyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.feature.extractor import BearingFeatureExtractor
from USTC.SSE.BearingPrediction.feature.engineering import FeatureConfig, SignalFeatureExtractor
from USTC.SSE.BearingPrediction.feature.backends import (
    CompositeFeatureBackend,
    FeatureBackend,
    FeatureBackendConfig,
    FeatureBackendInput,
    ManualFeatureBackend,
    TsfreshFeatureBackend,
    create_feature_backend,
)

__all__ = [
    "BearingFeatureExtractor",
    "CompositeFeatureBackend",
    "FeatureBackend",
    "FeatureBackendConfig",
    "FeatureBackendInput",
    "FeatureConfig",
    "ManualFeatureBackend",
    "SignalFeatureExtractor",
    "TsfreshFeatureBackend",
    "create_feature_backend",
]

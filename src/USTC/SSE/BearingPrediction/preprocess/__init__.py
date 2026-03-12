"""
Preprocess package

this file is for exposing preprocessing and stage partition components

created by cyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.preprocess.signal_processor import BearingSignalPreprocessor
from USTC.SSE.BearingPrediction.preprocess.pipeline import (
    PREPROCESSOR_REGISTRY,
    MinMaxNormalize,
    PreprocessingPipeline,
    RobustClip,
    SlidingWindowConfig,
    SlidingWindowSegmenter,
    ZScoreNormalize,
)
from USTC.SSE.BearingPrediction.preprocess.stage import (
    STAGE_STRATEGY_REGISTRY,
    DegradationStageResult,
    FPTStageStrategy,
    ThreeSigmaStageStrategy,
)

__all__ = [
    "BearingSignalPreprocessor",
    "DegradationStageResult",
    "FPTStageStrategy",
    "MinMaxNormalize",
    "PREPROCESSOR_REGISTRY",
    "PreprocessingPipeline",
    "RobustClip",
    "STAGE_STRATEGY_REGISTRY",
    "SlidingWindowConfig",
    "SlidingWindowSegmenter",
    "ThreeSigmaStageStrategy",
    "ZScoreNormalize",
]


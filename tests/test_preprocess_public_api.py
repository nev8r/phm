"""
Preprocess public api tests

this file is for guarding the supported preprocessing entry points

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import USTC.SSE.BearingPrediction.preprocess as preprocess


def test_preprocess_public_api_excludes_legacy_signal_preprocessor() -> None:
    assert "BearingSignalPreprocessor" not in preprocess.__all__
    assert not hasattr(preprocess, "BearingSignalPreprocessor")


def test_preprocess_public_api_keeps_configurable_pipeline_components() -> None:
    required_symbols = {
        "PreprocessingPipeline",
        "RobustClip",
        "ZScoreNormalize",
        "MinMaxNormalize",
        "SlidingWindowConfig",
        "SlidingWindowSegmenter",
    }

    assert required_symbols.issubset(set(preprocess.__all__))

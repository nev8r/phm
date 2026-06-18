"""
Skew processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

from numpy import ndarray

from scipy.stats import skew

from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class SkewProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Skew"

    def _compute(self, window: ndarray) -> float:
        return skew(window)

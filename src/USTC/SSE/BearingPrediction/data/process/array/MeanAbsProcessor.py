"""
Mean abs processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class MeanAbsProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Mean_abs"

    def _compute(self, window: ndarray) -> float:
        return np.mean(np.abs(window))

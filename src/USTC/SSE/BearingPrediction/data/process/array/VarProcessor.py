"""
Var processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class VarProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "Var"

    def _compute(self, window: ndarray) -> float:
        return np.var(window)

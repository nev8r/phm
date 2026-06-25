"""
Rms processor module

Purpose: load, label, or process bearing vibration data
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np
from numpy import ndarray

from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class RMSProcessor(WindowedProcessor):
    @property
    def name(self) -> str:
        return "RMS"

    def _compute(self, window: ndarray) -> float:
        return np.sqrt(np.mean(np.square(window)))

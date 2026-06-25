"""
Max processor module

Purpose: load, label, or process bearing vibration data
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np
from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class MaxProcessor(WindowedProcessor):
    @property
    def name(self):
        return "Max"

    def _compute(self, window):
        return np.max(window)

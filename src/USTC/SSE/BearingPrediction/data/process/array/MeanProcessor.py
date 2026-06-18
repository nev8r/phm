"""
Mean processor module

this file is for processing bearing vibration signals and features

created by cyj

copyright USTC

2026
"""

import numpy as np
from USTC.SSE.BearingPrediction.data.process.array.WindowedProcessor import WindowedProcessor


class MeanProcessor(WindowedProcessor):
    @property
    def name(self):
        return "Mean"

    def _compute(self, window):
        return np.mean(window)

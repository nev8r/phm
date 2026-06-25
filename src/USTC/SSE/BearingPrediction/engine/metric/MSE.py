"""
MSE metric module

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import numpy as np

from USTC.SSE.BearingPrediction.data.Dataset import Dataset
from USTC.SSE.BearingPrediction.engine.Result import Result
from USTC.SSE.BearingPrediction.engine.metric.ABCMetric import ABCMetric


class MSE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MSE'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y
        return float(np.mean((r_hat - r) ** 2, axis=0))

"""
MAE metric module

this file is for computing mae evaluation metrics

created by zdh

copyright USTC

2026
"""

import numpy as np

from USTC.SSE.BearingPrediction.data.Dataset import Dataset
from USTC.SSE.BearingPrediction.engine.Result import Result
from USTC.SSE.BearingPrediction.engine.metric.ABCMetric import ABCMetric


class MAE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAE'

    @property
    def is_higher_better(self) -> bool:
        return False

    def value(self, test_set: Dataset, result: Result) -> float:
        r_hat = result.y_hat
        r = test_set.y
        return float(np.mean(np.abs(r - r_hat)))

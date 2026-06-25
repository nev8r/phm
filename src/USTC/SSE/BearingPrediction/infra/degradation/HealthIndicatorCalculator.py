"""
Health indicator calculator interface.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.degradation.HealthIndicatorFrame import HealthIndicatorFrame


class HealthIndicatorCalculator(ABC):
    @abstractmethod
    def calculate(self, features: pd.DataFrame) -> HealthIndicatorFrame:
        raise NotImplementedError

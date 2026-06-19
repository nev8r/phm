"""
Health indicator calculator interface.
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.degradation.HealthIndicatorFrame import HealthIndicatorFrame


class HealthIndicatorCalculator(ABC):
    @abstractmethod
    def calculate(self, features: pd.DataFrame) -> HealthIndicatorFrame:
        raise NotImplementedError

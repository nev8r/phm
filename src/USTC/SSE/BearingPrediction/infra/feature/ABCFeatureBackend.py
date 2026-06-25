"""
Feature backend interface.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FeatureFrame


class ABCFeatureBackend(ABC):
    @abstractmethod
    def extract(self, index: pd.DataFrame) -> FeatureFrame:
        raise NotImplementedError

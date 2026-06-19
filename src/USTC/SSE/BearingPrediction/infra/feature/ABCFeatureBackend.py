"""
Feature backend interface.
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FeatureFrame


class ABCFeatureBackend(ABC):
    @abstractmethod
    def extract(self, index: pd.DataFrame) -> FeatureFrame:
        raise NotImplementedError

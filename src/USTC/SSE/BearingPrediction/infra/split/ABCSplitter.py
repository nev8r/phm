"""
Splitter interface.
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult


class ABCSplitter(ABC):
    @abstractmethod
    def split(self, index: pd.DataFrame) -> SplitResult:
        raise NotImplementedError

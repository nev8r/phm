"""
Splitter interface.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from abc import ABC, abstractmethod

import pandas as pd

from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult


class ABCSplitter(ABC):
    @abstractmethod
    def split(self, index: pd.DataFrame) -> SplitResult:
        raise NotImplementedError

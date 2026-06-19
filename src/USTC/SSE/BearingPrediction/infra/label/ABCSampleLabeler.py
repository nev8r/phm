"""
Abstract sample labeler.
"""

from abc import ABC, abstractmethod

import pandas as pd


class ABCSampleLabeler(ABC):
    @abstractmethod
    def label(self, index: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError

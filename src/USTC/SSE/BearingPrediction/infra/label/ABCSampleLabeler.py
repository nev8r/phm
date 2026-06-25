"""
Abstract sample labeler.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from abc import ABC, abstractmethod

import pandas as pd


class ABCSampleLabeler(ABC):
    @abstractmethod
    def label(self, index: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError

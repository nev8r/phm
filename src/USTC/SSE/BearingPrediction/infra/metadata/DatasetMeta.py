"""
Dataset metadata interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable

from USTC.SSE.BearingPrediction.infra.metadata.BearingMeta import BearingMeta


class DatasetMeta(ABC):
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_bearing_meta(self, bearing_id: str) -> BearingMeta:
        raise NotImplementedError

    @abstractmethod
    def iter_bearing_meta(self) -> Iterable[BearingMeta]:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "bearings": [bearing.to_dict() for bearing in self.iter_bearing_meta()],
        }

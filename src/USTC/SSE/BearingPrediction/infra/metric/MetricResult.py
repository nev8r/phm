"""
Metric result container.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class MetricResult:
    split: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)

"""
Metric result container.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class MetricResult:
    split: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)

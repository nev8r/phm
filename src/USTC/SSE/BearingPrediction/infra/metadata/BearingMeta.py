"""
Bearing-level metadata.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

from USTC.SSE.BearingPrediction.infra.metadata.BearingGeometry import BearingGeometry


@dataclass(frozen=True)
class BearingMeta:
    dataset: str
    bearing_id: str
    condition_id: str
    sampling_rate: float
    sample_interval_seconds: float
    expected_points_per_sample: int
    channels: Tuple[str, ...]
    speed_hz: Optional[float] = None
    load_n: Optional[float] = None
    lifetime_samples: Optional[int] = None
    lifetime_seconds: Optional[float] = None
    fault_element: Optional[Tuple[str, ...]] = None
    source_group: Optional[str] = None
    geometry: Optional[BearingGeometry] = None

    def to_dict(self) -> Dict:
        return asdict(self)

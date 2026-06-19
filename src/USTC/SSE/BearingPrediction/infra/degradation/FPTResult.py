"""
First prediction time result.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class FPTResult:
    dataset: str
    bearing_id: str
    condition_id: str
    fpt_index: int
    fpt_sample_uid: str
    fpt_timestep: int
    threshold: float
    method: str
    success: bool
    fallback_used: bool
    params: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

"""
Bearing geometry metadata.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BearingGeometry:
    ball_count: Optional[int] = None
    ball_diameter_mm: Optional[float] = None
    pitch_diameter_mm: Optional[float] = None
    contact_angle_deg: Optional[float] = None

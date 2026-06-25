"""
Bearing geometry metadata.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BearingGeometry:
    ball_count: Optional[int] = None
    ball_diameter_mm: Optional[float] = None
    pitch_diameter_mm: Optional[float] = None
    contact_angle_deg: Optional[float] = None

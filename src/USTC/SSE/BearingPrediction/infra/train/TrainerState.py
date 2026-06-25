"""
Trainer state.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    should_stop: bool = False

    def to_dict(self):
        return asdict(self)

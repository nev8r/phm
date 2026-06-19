"""
Trainer state.
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

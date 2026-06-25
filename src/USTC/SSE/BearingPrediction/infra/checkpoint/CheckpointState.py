"""
Checkpoint metadata schema.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    global_step: int
    best_metric: Optional[float]
    best_epoch: Optional[int]
    model_spec: Dict[str, Any]
    task_spec: Dict[str, Any]
    trainer_config: Dict[str, Any]
    feature_columns: List[str]
    target_columns: List[str]
    history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""
Training context container.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainContext:
    run_dir: Path
    artifacts: Any

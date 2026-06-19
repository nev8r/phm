"""
Training context container.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainContext:
    run_dir: Path
    artifacts: Any

"""
Model specification.

Purpose: define model components for bearing PHM tasks
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict

import hashlib
import json


@dataclass(frozen=True)
class ModelSpec:
    name: str
    class_name: str
    input_dim: int
    output_dim: int
    params: Dict[str, Any]
    input_mode: str
    task_type: str
    created_by: str = "Stage5 ModelFactory"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_by", None)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

"""
Task specification.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import hashlib
import json


@dataclass(frozen=True)
class TaskSpec:
    name: str
    version: str
    task_type: str
    input_mode: str
    feature_source: str
    feature_columns: List[str]
    target_columns: List[str]
    sequence: Optional[Dict[str, Any]] = None
    created_by: str = "Stage4 TaskBuilder"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_by", None)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

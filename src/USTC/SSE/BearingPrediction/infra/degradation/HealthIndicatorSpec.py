"""
Health indicator spec metadata.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HealthIndicatorSpec:
    method: str
    params: Dict[str, Any]
    version: str = "v1"
    created_by: str = "Stage3 HealthIndicatorCalculator"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

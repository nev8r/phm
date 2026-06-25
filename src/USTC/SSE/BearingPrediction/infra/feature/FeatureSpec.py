"""
Feature spec metadata.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    backends: List[Dict[str, Any]]
    cleaner: Dict[str, Any]
    version: str = "v1"
    created_by: str = "Stage2 FeatureExtractor"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

"""
Label specification.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import hashlib
import json


@dataclass(frozen=True)
class LabelSpec:
    name: str
    version: str
    requires_features: bool
    outputs: List[Dict[str, Any]]
    hi: Optional[Dict[str, Any]] = None
    fpt: Optional[Dict[str, Any]] = None
    created_by: str = "Stage3 LabelBuilder"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_by", None)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

"""
Analysis specification.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List

import hashlib
import json


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    version: str
    feature_source: str
    fit_scope: str
    enabled_sections: List[str]
    created_by: str = "Stage6 AnalysisBuilder"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["hash"] = self.hash()
        return data

    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_by", None)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

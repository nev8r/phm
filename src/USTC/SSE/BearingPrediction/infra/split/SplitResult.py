"""
Split result object.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SplitResult:
    name: str
    train_sample_uids: List[str]
    val_sample_uids: List[str]
    test_sample_uids: List[str]
    train_bearings: List[str]
    val_bearings: List[str]
    test_bearings: List[str]
    split_spec: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def report(self) -> Dict:
        checks = [
            _check("no_sample_overlap", _is_disjoint(self.train_sample_uids, self.val_sample_uids, self.test_sample_uids)),
            _check("no_bearing_overlap", _is_disjoint(self.train_bearings, self.val_bearings, self.test_bearings)),
            _check("non_empty_train", len(self.train_sample_uids) > 0),
            _check("non_empty_test", len(self.test_sample_uids) > 0),
        ]
        if self.val_bearings:
            checks.append(_check("non_empty_val", len(self.val_sample_uids) > 0))

        return {
            "ok": all(check["ok"] for check in checks),
            "name": self.name,
            "num_train_samples": len(self.train_sample_uids),
            "num_val_samples": len(self.val_sample_uids),
            "num_test_samples": len(self.test_sample_uids),
            "train_bearings": self.train_bearings,
            "val_bearings": self.val_bearings,
            "test_bearings": self.test_bearings,
            "checks": checks,
        }


def _check(name: str, ok: bool) -> Dict:
    return {"name": name, "ok": bool(ok)}


def _is_disjoint(*groups: List[str]) -> bool:
    seen = set()
    for group in groups:
        values = set(group)
        if seen.intersection(values):
            return False
        seen.update(values)
    return True

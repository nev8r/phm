"""
Sample index validator.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from USTC.SSE.BearingPrediction.infra.index.SampleIndex import SAMPLE_INDEX_COLUMNS


class IndexValidator:
    """
    Validate sample-index schema and sample-level consistency.
    """

    def validate(self, index: pd.DataFrame, strict: bool = True) -> Dict[str, Any]:
        checks = [
            _check("required_columns_present", all(column in index.columns for column in SAMPLE_INDEX_COLUMNS)),
            _check("non_empty", len(index) > 0),
        ]

        if not checks[0]["ok"]:
            return self._finish(index, checks, strict)

        checks.extend([
            _check("sample_uid_unique", index["sample_uid"].is_unique),
            _check("file_path_exists", index["file_path"].map(lambda value: Path(str(value)).exists()).all()),
            _check("no_null_dataset", _has_no_null_or_empty(index["dataset"])),
            _check("no_null_bearing_id", _has_no_null_or_empty(index["bearing_id"])),
            _check("no_null_sample_id", _has_no_null_or_empty(index["sample_id"])),
            _check("timestep_non_negative", (index["timestep"] >= 0).all()),
            _check("timestep_monotonic_within_bearing", _is_timestep_monotonic(index)),
            _check("expected_points_positive", (index["expected_points"] > 0).all()),
            _check("sampling_rate_positive", (index["sampling_rate"] > 0).all()),
            _check("no_duplicate_bearing_sample", not index.duplicated(["dataset", "bearing_id", "sample_id"]).any()),
        ])

        return self._finish(index, checks, strict)

    def _finish(self, index: pd.DataFrame, checks: List[Dict[str, Any]], strict: bool) -> Dict[str, Any]:
        ok = all(check["ok"] for check in checks)
        report = {
            "ok": ok,
            "dataset": _single_value(index, "dataset"),
            "num_rows": int(len(index)),
            "num_bearings": int(index["bearing_id"].nunique()) if "bearing_id" in index else 0,
            "checks": checks,
        }
        if strict and not ok:
            failed = ", ".join(check["name"] for check in checks if not check["ok"])
            raise ValueError(f"Index validation failed: {failed}")
        return report


def _check(name: str, ok: bool, message: str = "") -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "message": message}


def _has_no_null_or_empty(series: pd.Series) -> bool:
    return series.notna().all() and series.map(lambda value: str(value).strip() != "").all()


def _is_timestep_monotonic(index: pd.DataFrame) -> bool:
    for _, group in index.groupby(["dataset", "bearing_id"], dropna=False):
        timesteps = list(group["timestep"])
        if timesteps != sorted(timesteps):
            return False
    return True


def _single_value(index: pd.DataFrame, column: str):
    if column not in index or index.empty:
        return None
    values = index[column].dropna().unique()
    return values[0] if len(values) == 1 else None

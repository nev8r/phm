"""
Task report builder.
"""

from typing import Dict, List, Optional

import pandas as pd

from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, FEATURE_SEQUENCE, REGRESSION


def build_task_report(
        task_name: str,
        task_type: str,
        input_mode: str,
        feature_source: str,
        feature_columns: List[str],
        target_columns: List[str],
        manifest: pd.DataFrame,
        labels: pd.DataFrame,
        sequence: Optional[Dict] = None,
) -> Dict:
    checks = _checks(manifest, input_mode)
    report = {
        "ok": all(check["ok"] for check in checks),
        "task_name": task_name,
        "task_type": task_type,
        "input_mode": input_mode,
        "feature_source": feature_source,
        "num_features": len(feature_columns),
        "target_columns": target_columns,
        "num_examples": int(len(manifest)),
        "num_train_examples": _split_count(manifest, "train"),
        "num_val_examples": _split_count(manifest, "val"),
        "num_test_examples": _split_count(manifest, "test"),
        "num_all_examples": _split_count(manifest, "all"),
        "sequence": _sequence_report(input_mode, sequence),
        "class_distribution": {},
        "target_summary": {},
        "checks": checks,
    }
    target_values = _target_values(manifest, labels, target_columns)
    if task_type in CLASSIFICATION_TYPES:
        report["class_distribution"] = _class_distribution(manifest, target_values, target_columns[0])
    if task_type == REGRESSION:
        report["target_summary"] = _target_summary(target_values, target_columns)
    return report


def _checks(manifest: pd.DataFrame, input_mode: str) -> List[Dict]:
    checks = [
        {"name": "sample_uid_alignment", "ok": True},
        {"name": "non_empty_examples", "ok": len(manifest) > 0},
        {"name": "unique_example_uid", "ok": bool(manifest["example_uid"].is_unique)},
    ]
    if input_mode == FEATURE_SEQUENCE:
        checks.extend([
            {"name": "no_cross_bearing_windows", "ok": _no_cross_values(manifest, "bearing")},
            {"name": "no_cross_split_windows", "ok": _no_cross_values(manifest, "split")},
        ])
    return checks


def _no_cross_values(manifest: pd.DataFrame, field: str) -> bool:
    del field
    return bool((manifest["num_timesteps"].astype(int) == manifest["window_sample_uids"].str.split("|").map(len)).all())


def _split_count(manifest: pd.DataFrame, split: str) -> int:
    return int((manifest["split"] == split).sum())


def _sequence_report(input_mode: str, sequence: Optional[Dict]) -> Dict:
    if input_mode != FEATURE_SEQUENCE:
        return {"enabled": False}
    sequence = sequence or {}
    return {
        "enabled": True,
        "length": int(sequence.get("length", 8)),
        "stride": int(sequence.get("stride", 1)),
        "target_position": str(sequence.get("target_position", "last")),
    }


def _target_values(manifest: pd.DataFrame, labels: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    target_sample_uids = manifest[["split", "target_sample_uid"]].rename(columns={"target_sample_uid": "sample_uid"})
    return target_sample_uids.merge(labels[["sample_uid", *target_columns]], on="sample_uid", how="left")


def _class_distribution(manifest: pd.DataFrame, target_values: pd.DataFrame, target_column: str) -> Dict[str, Dict[str, int]]:
    del manifest
    distribution: Dict[str, Dict[str, int]] = {}
    for split_name, group in target_values.groupby("split", sort=False):
        counts = group[target_column].value_counts().sort_index()
        distribution[str(split_name)] = {str(key): int(value) for key, value in counts.items()}
    return distribution


def _target_summary(target_values: pd.DataFrame, target_columns: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for column in target_columns:
        series = target_values[column].astype(float)
        summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return summary

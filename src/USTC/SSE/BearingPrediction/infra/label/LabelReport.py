"""
Label report builder.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List, Optional

import pandas as pd

from USTC.SSE.BearingPrediction.infra.label.LabelFrame import LABEL_INDEX_COLUMNS


def build_label_report(
        labels: pd.DataFrame,
        label_set: str,
        requires_features: bool,
        fpt_payload: Optional[Dict] = None,
        fault_type_stage_mapping: Optional[Dict[str, int]] = None,
) -> Dict:
    label_columns = [column for column in labels.columns if column not in LABEL_INDEX_COLUMNS]
    rul_checks = _rul_checks(labels)
    checks = [*rul_checks]
    report = {
        "ok": all(check["ok"] for check in checks),
        "label_set": label_set,
        "num_samples": int(len(labels)),
        "num_bearings": int(labels[["dataset", "bearing_id"]].drop_duplicates().shape[0]),
        "label_columns": label_columns,
        "requires_features": bool(requires_features),
        "hi_enabled": fpt_payload is not None,
        "fpt_enabled": fpt_payload is not None,
        "num_fpt_success": _count_fpt(fpt_payload, success=True),
        "num_fpt_fallback": _count_fpt(fpt_payload, fallback=True),
        "health_state_distribution": _distribution(labels, "health_state_name"),
        "early_fault_distribution": _distribution(labels, "early_fault"),
        "rul_checks": rul_checks,
        "health_state_is_pseudo_label": "health_state_id" in labels.columns,
    }
    if fault_type_stage_mapping:
        report["fault_type_stage_mapping"] = fault_type_stage_mapping
        report["fault_type_stage_distribution"] = _distribution(labels, "fault_type_stage_name")
    return report


def _count_fpt(fpt_payload: Optional[Dict], success: bool = False, fallback: bool = False) -> int:
    if not fpt_payload:
        return 0
    count = 0
    for result in fpt_payload.get("results", []):
        if success and bool(result.get("success")):
            count += 1
        if fallback and bool(result.get("fallback_used")):
            count += 1
    return count


def _distribution(labels: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in labels.columns:
        return {}
    counts = labels[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _rul_checks(labels: pd.DataFrame) -> List[Dict]:
    checks: List[Dict] = []
    if "linear_rul_norm" in labels.columns:
        checks.append({
            "name": "linear_rul_within_0_1",
            "ok": bool(labels["linear_rul_norm"].between(0, 1).all()),
        })
        checks.append({
            "name": "linear_rul_ends_at_zero",
            "ok": _last_values_are_zero(labels, "linear_rul_norm"),
        })
    if "piecewise_rul_norm" in labels.columns:
        checks.append({
            "name": "piecewise_rul_within_0_1",
            "ok": bool(labels["piecewise_rul_norm"].between(0, 1).all()),
        })
        checks.append({
            "name": "piecewise_rul_ends_at_zero",
            "ok": _last_values_are_zero(labels, "piecewise_rul_norm"),
        })
    return checks


def _last_values_are_zero(labels: pd.DataFrame, column: str) -> bool:
    for _, group in labels.groupby(["dataset", "bearing_id"], sort=False):
        value = float(group.sort_values("timestep").iloc[-1][column])
        if abs(value) > 1.0e-9:
            return False
    return True

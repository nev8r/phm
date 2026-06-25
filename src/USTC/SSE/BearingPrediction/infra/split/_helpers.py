"""
Shared splitter helpers.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Iterable, List

import pandas as pd


def sorted_unique(values: Iterable) -> List[str]:
    return sorted([str(value) for value in pd.Series(list(values)).dropna().unique()], key=_natural_key)


def sample_uids_for_bearings(index: pd.DataFrame, bearings: List[str]) -> List[str]:
    if not bearings:
        return []
    return list(index[index["bearing_id"].isin(bearings)]["sample_uid"])


def sample_uids_for_conditions(index: pd.DataFrame, condition_ids: List[str]) -> List[str]:
    if not condition_ids:
        return []
    return list(index[index["condition_id"].isin(condition_ids)]["sample_uid"])


def validate_report(result):
    report = result.report()
    if not report["ok"]:
        failed = ", ".join(check["name"] for check in report["checks"] if not check["ok"])
        raise ValueError(f"Split validation failed: {failed}")
    return result


def _natural_key(value: str):
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]

"""
XJTU-SY splitter that groups all conditions by bearing suffix index.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import re
from typing import Iterable, List, Optional, Set

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.split.ABCSplitter import ABCSplitter
from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult
from USTC.SSE.BearingPrediction.infra.split._helpers import sample_uids_for_bearings, sorted_unique, validate_report


class BearingIndexSplitter(ABCSplitter):
    """
    Split XJTU-SY across operating conditions by the suffix in BearingX_Y.

    Example:
        train indices [1, 2, 3] select Bearing1_1, Bearing1_2, Bearing1_3,
        Bearing2_1, ..., Bearing3_3 when all three conditions are present.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def split(self, index: pd.DataFrame) -> SplitResult:
        train_indices = _int_set(OmegaConf.select(self.cfg, "train_bearing_indices", default=[1, 2, 3]))
        val_indices = _int_set(OmegaConf.select(self.cfg, "val_bearing_indices", default=[4]))
        test_indices = _int_set(OmegaConf.select(self.cfg, "test_bearing_indices", default=[5]))
        _ensure_disjoint_indices(train_indices, val_indices, test_indices)

        condition_ids = _condition_filter(OmegaConf.select(self.cfg, "condition_ids", default=[]))
        subset = index.copy()
        if condition_ids is not None:
            subset = subset[subset["condition_id"].isin(condition_ids)]

        bearings = sorted_unique(subset["bearing_id"])
        train_bearings = _bearings_for_indices(bearings, train_indices)
        val_bearings = _bearings_for_indices(bearings, val_indices)
        test_bearings = _bearings_for_indices(bearings, test_indices)

        result = SplitResult(
            name="xjtu_bearing_index_split",
            train_sample_uids=sample_uids_for_bearings(subset, train_bearings),
            val_sample_uids=sample_uids_for_bearings(subset, val_bearings),
            test_sample_uids=sample_uids_for_bearings(subset, test_bearings),
            train_bearings=train_bearings,
            val_bearings=val_bearings,
            test_bearings=test_bearings,
            split_spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        return validate_report(result)


def _int_set(values: Iterable) -> Set[int]:
    return {int(value) for value in values}


def _condition_filter(values: Iterable) -> Optional[List[str]]:
    condition_ids = [str(value) for value in values]
    return condition_ids or None


def _ensure_disjoint_indices(*groups: Set[int]) -> None:
    seen: Set[int] = set()
    for group in groups:
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"bearing index groups must be disjoint, overlap={sorted(overlap)}")
        seen.update(group)


def _bearings_for_indices(bearings: List[str], indices: Set[int]) -> List[str]:
    return [bearing for bearing in bearings if _bearing_suffix_index(bearing) in indices]


def _bearing_suffix_index(bearing_id: str) -> int:
    match = re.search(r"_(\d+)$", str(bearing_id))
    if match is None:
        raise ValueError(f"Cannot parse bearing suffix index from bearing_id={bearing_id}")
    return int(match.group(1))

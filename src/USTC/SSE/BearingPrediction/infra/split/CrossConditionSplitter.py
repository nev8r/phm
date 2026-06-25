"""
Cross-condition splitter for XJTU-SY.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.split.ABCSplitter import ABCSplitter
from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult
from USTC.SSE.BearingPrediction.infra.split._helpers import sample_uids_for_conditions, sorted_unique, validate_report


class CrossConditionSplitter(ABCSplitter):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def split(self, index: pd.DataFrame) -> SplitResult:
        train_conditions = list(OmegaConf.select(self.cfg, "train_condition_ids", default=[]))
        val_conditions = list(OmegaConf.select(self.cfg, "val_condition_ids", default=[]))
        test_conditions = list(OmegaConf.select(self.cfg, "test_condition_ids", default=[]))

        _ensure_disjoint_conditions(train_conditions, val_conditions, test_conditions)

        train_subset = index[index["condition_id"].isin(train_conditions)]
        val_subset = index[index["condition_id"].isin(val_conditions)]
        test_subset = index[index["condition_id"].isin(test_conditions)]

        result = SplitResult(
            name="xjtu_cross_condition",
            train_sample_uids=sample_uids_for_conditions(index, train_conditions),
            val_sample_uids=sample_uids_for_conditions(index, val_conditions),
            test_sample_uids=sample_uids_for_conditions(index, test_conditions),
            train_bearings=sorted_unique(train_subset["bearing_id"]),
            val_bearings=sorted_unique(val_subset["bearing_id"]),
            test_bearings=sorted_unique(test_subset["bearing_id"]),
            split_spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        return validate_report(result)


def _ensure_disjoint_conditions(*groups):
    seen = set()
    for group in groups:
        current = set(group)
        overlap = seen.intersection(current)
        if overlap:
            raise ValueError(f"condition groups must be disjoint, overlap={sorted(overlap)}")
        seen.update(current)

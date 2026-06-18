"""
Leave-one-bearing-out splitter for XJTU-SY.
"""

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.split.ABCSplitter import ABCSplitter
from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult
from USTC.SSE.BearingPrediction.infra.split._helpers import sample_uids_for_bearings, sorted_unique, validate_report


class LeaveOneBearingOutSplitter(ABCSplitter):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def split(self, index: pd.DataFrame) -> SplitResult:
        condition_id = str(OmegaConf.select(self.cfg, "condition_id", default=""))
        test_bearing = str(OmegaConf.select(self.cfg, "test_bearing_id", default=""))
        val_bearing = str(OmegaConf.select(self.cfg, "val_bearing_id", default=""))
        if not condition_id or not test_bearing or not val_bearing:
            raise ValueError("condition_id, test_bearing_id, and val_bearing_id are required")

        subset = index[index["condition_id"] == condition_id]
        bearings = sorted_unique(subset["bearing_id"])
        if test_bearing not in bearings:
            raise ValueError(f"test_bearing_id not found in condition {condition_id}: {test_bearing}")
        if val_bearing not in bearings:
            raise ValueError(f"val_bearing_id not found in condition {condition_id}: {val_bearing}")
        if test_bearing == val_bearing:
            raise ValueError("test_bearing_id and val_bearing_id must be different")

        train_bearings = [bearing for bearing in bearings if bearing not in {test_bearing, val_bearing}]
        result = SplitResult(
            name="xjtu_leave_one_bearing_out",
            train_sample_uids=sample_uids_for_bearings(subset, train_bearings),
            val_sample_uids=sample_uids_for_bearings(subset, [val_bearing]),
            test_sample_uids=sample_uids_for_bearings(subset, [test_bearing]),
            train_bearings=train_bearings,
            val_bearings=[val_bearing],
            test_bearings=[test_bearing],
            split_spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        return validate_report(result)

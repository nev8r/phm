"""
Official PHM2012 splitter.
"""

from typing import List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.split.ABCSplitter import ABCSplitter
from USTC.SSE.BearingPrediction.infra.split.SplitResult import SplitResult
from USTC.SSE.BearingPrediction.infra.split._helpers import sample_uids_for_bearings, sorted_unique, validate_report


class OfficialPHM2012Splitter(ABCSplitter):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def split(self, index: pd.DataFrame) -> SplitResult:
        mode = str(OmegaConf.select(self.cfg, "mode", default="source_group"))
        if mode == "source_group":
            return self._split_by_source_group(index)
        if mode == "explicit":
            return self._split_explicit(index)
        raise ValueError(f"Unsupported PHM2012 split mode: {mode}")

    def _split_by_source_group(self, index: pd.DataFrame) -> SplitResult:
        train_source_group = str(OmegaConf.select(self.cfg, "train_source_group", default="Learning_set"))
        test_source_group = str(OmegaConf.select(self.cfg, "test_source_group", default="Full_Test_Set"))
        val_bearings = list(OmegaConf.select(self.cfg, "val_bearings", default=[]))

        train_source = index[index["source_group"] == train_source_group]
        test_source = index[index["source_group"] == test_source_group]
        train_bearings = [
            bearing for bearing in sorted_unique(train_source["bearing_id"])
            if bearing not in set(val_bearings)
        ]
        test_bearings = sorted_unique(test_source["bearing_id"])

        result = SplitResult(
            name="phm2012_official",
            train_sample_uids=sample_uids_for_bearings(train_source, train_bearings),
            val_sample_uids=sample_uids_for_bearings(train_source, val_bearings),
            test_sample_uids=list(test_source["sample_uid"]),
            train_bearings=train_bearings,
            val_bearings=val_bearings,
            test_bearings=test_bearings,
            split_spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        return validate_report(result)

    def _split_explicit(self, index: pd.DataFrame) -> SplitResult:
        train_bearings = _list_cfg(self.cfg, "train_bearings")
        val_bearings = _list_cfg(self.cfg, "val_bearings")
        test_bearings = _list_cfg(self.cfg, "test_bearings")
        result = SplitResult(
            name="phm2012_official",
            train_sample_uids=sample_uids_for_bearings(index, train_bearings),
            val_sample_uids=sample_uids_for_bearings(index, val_bearings),
            test_sample_uids=sample_uids_for_bearings(index, test_bearings),
            train_bearings=train_bearings,
            val_bearings=val_bearings,
            test_bearings=test_bearings,
            split_spec=OmegaConf.to_container(self.cfg, resolve=True),
        )
        return validate_report(result)


def _list_cfg(cfg: DictConfig, key: str) -> List[str]:
    return list(OmegaConf.select(cfg, key, default=[]))

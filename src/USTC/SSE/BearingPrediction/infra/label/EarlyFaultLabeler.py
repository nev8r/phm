"""
Early fault binary labeler.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.label.ABCSampleLabeler import ABCSampleLabeler


class EarlyFaultLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, fpt_map: Dict[Tuple[str, str], Dict]) -> pd.DataFrame:
        normal_value = int(OmegaConf.select(self.cfg, "normal_value", default=0))
        abnormal_value = int(OmegaConf.select(self.cfg, "abnormal_value", default=1))
        rows: List[Dict] = []
        for key, group in _groups(index):
            fpt_index = _get_fpt_index(fpt_map, key)
            for position, row in group.iterrows():
                rows.append({
                    "sample_uid": row["sample_uid"],
                    "early_fault": normal_value if position < fpt_index else abnormal_value,
                })
        return pd.DataFrame(rows)


def _groups(index: pd.DataFrame):
    for key, group in index.groupby(["dataset", "bearing_id"], sort=False):
        yield key, group.sort_values("timestep").reset_index(drop=True)


def _get_fpt_index(fpt_map: Dict[Tuple[str, str], Dict], key: Tuple[str, str]) -> int:
    if key not in fpt_map:
        raise ValueError(f"Missing FPT result for {key}")
    return int(fpt_map[key]["fpt_index"])

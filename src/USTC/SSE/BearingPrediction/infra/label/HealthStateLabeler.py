"""
Health state pseudo-labeler.

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


class HealthStateLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, fpt_map: Dict[Tuple[str, str], Dict]) -> pd.DataFrame:
        state_names = _state_names(self.cfg)
        boundaries = [float(value) for value in OmegaConf.select(self.cfg, "post_fpt_boundaries", default=[0.4, 0.8])]
        rows: List[Dict] = []

        for key, group in _groups(index):
            fpt_index = _get_fpt_index(fpt_map, key)
            total = len(group)
            denominator = max(total - 1 - fpt_index, 1)
            for position, row in group.iterrows():
                if position < fpt_index:
                    state_id = 0
                else:
                    progress = (position - fpt_index) / denominator
                    state_id = 1
                    for boundary in boundaries:
                        if progress >= boundary:
                            state_id += 1
                rows.append({
                    "sample_uid": row["sample_uid"],
                    "health_state_id": int(state_id),
                    "health_state_name": state_names.get(int(state_id), str(state_id)),
                })
        return pd.DataFrame(rows)


def _state_names(cfg: DictConfig) -> Dict[int, str]:
    raw = OmegaConf.to_container(OmegaConf.select(cfg, "state_names", default={
        0: "healthy",
        1: "slight",
        2: "moderate",
        3: "severe",
    }), resolve=True)
    return {int(key): str(value) for key, value in raw.items()}


def _groups(index: pd.DataFrame):
    for key, group in index.groupby(["dataset", "bearing_id"], sort=False):
        yield key, group.sort_values("timestep").reset_index(drop=True)


def _get_fpt_index(fpt_map: Dict[Tuple[str, str], Dict], key: Tuple[str, str]) -> int:
    if key not in fpt_map:
        raise ValueError(f"Missing FPT result for {key}")
    return int(fpt_map[key]["fpt_index"])

"""
Piecewise RUL labeler.

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


class PiecewiseRulLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, fpt_map: Dict[Tuple[str, str], Dict]) -> pd.DataFrame:
        rows: List[Dict] = []
        output_steps = bool(OmegaConf.select(self.cfg, "output_steps", default=True))
        output_seconds = bool(OmegaConf.select(self.cfg, "output_seconds", default=True))
        normalize = bool(OmegaConf.select(self.cfg, "normalize", default=True))

        for key, group in _groups(index):
            fpt_index = _get_fpt_index(fpt_map, key)
            total = len(group)
            denominator = max(total - 1 - fpt_index, 1)
            for position, row in group.iterrows():
                if position < fpt_index:
                    rul_norm = 1.0
                else:
                    rul_norm = (total - 1 - position) / denominator
                rul_norm = float(max(0.0, min(1.0, rul_norm)))
                rul_steps = float(rul_norm * denominator)
                output = {"sample_uid": row["sample_uid"]}
                if output_steps:
                    output["piecewise_rul_steps"] = rul_steps
                if output_seconds:
                    output["piecewise_rul_seconds"] = float(rul_steps * _sample_interval(row))
                if normalize:
                    output["piecewise_rul_norm"] = rul_norm
                rows.append(output)
        return pd.DataFrame(rows)


def _groups(index: pd.DataFrame):
    for key, group in index.groupby(["dataset", "bearing_id"], sort=False):
        yield key, group.sort_values("timestep").reset_index(drop=True)


def _get_fpt_index(fpt_map: Dict[Tuple[str, str], Dict], key: Tuple[str, str]) -> int:
    if key not in fpt_map:
        raise ValueError(f"Missing FPT result for {key}")
    return int(fpt_map[key]["fpt_index"])


def _sample_interval(row) -> float:
    return float(row.get("sample_interval_seconds", 1.0) or 1.0)

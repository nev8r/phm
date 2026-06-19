"""
Capped RUL labeler.
"""

from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.label.ABCSampleLabeler import ABCSampleLabeler


class CappedRulLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        max_rul_steps = int(OmegaConf.select(self.cfg, "max_rul_steps", default=125))
        if max_rul_steps <= 0:
            raise ValueError("max_rul_steps must be > 0")
        output_steps = bool(OmegaConf.select(self.cfg, "output_steps", default=True))
        output_seconds = bool(OmegaConf.select(self.cfg, "output_seconds", default=True))
        normalize = bool(OmegaConf.select(self.cfg, "normalize", default=True))
        rows: List[Dict] = []

        for _, group in index.groupby(["dataset", "bearing_id"], sort=False):
            group = group.sort_values("timestep").reset_index(drop=True)
            total = len(group)
            for position, row in group.iterrows():
                capped_steps = int(min(total - 1 - position, max_rul_steps))
                output = {"sample_uid": row["sample_uid"]}
                if output_steps:
                    output["capped_rul_steps"] = capped_steps
                if output_seconds:
                    output["capped_rul_seconds"] = int(capped_steps * _sample_interval(row))
                if normalize:
                    output["capped_rul_norm"] = float(capped_steps / max_rul_steps)
                rows.append(output)
        return pd.DataFrame(rows)


def _sample_interval(row) -> float:
    return float(row.get("sample_interval_seconds", 1.0) or 1.0)

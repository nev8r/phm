"""
Linear RUL labeler.
"""

from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.label.ABCSampleLabeler import ABCSampleLabeler


class LinearRulLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        rows: List[Dict] = []
        output_steps = bool(OmegaConf.select(self.cfg, "output_steps", default=True))
        output_seconds = bool(OmegaConf.select(self.cfg, "output_seconds", default=True))
        normalize = bool(OmegaConf.select(self.cfg, "normalize", default=True))

        for _, group in index.groupby(["dataset", "bearing_id"], sort=False):
            group = group.sort_values("timestep").reset_index(drop=True)
            total = len(group)
            denominator = max(total - 1, 1)
            for position, row in group.iterrows():
                rul_steps = int(total - 1 - position)
                output = {"sample_uid": row["sample_uid"]}
                if output_steps:
                    output["linear_rul_steps"] = rul_steps
                if output_seconds:
                    output["linear_rul_seconds"] = int(rul_steps * _sample_interval(row))
                if normalize:
                    output["linear_rul_norm"] = float(rul_steps / denominator)
                rows.append(output)
        return pd.DataFrame(rows)


def _sample_interval(row) -> float:
    return float(row.get("sample_interval_seconds", 1.0) or 1.0)

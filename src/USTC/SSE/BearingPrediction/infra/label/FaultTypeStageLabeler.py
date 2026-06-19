"""
Fault type stage labeler for datasets with final fault metadata.
"""

from typing import Dict, List, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.label.ABCSampleLabeler import ABCSampleLabeler


class FaultTypeStageLabeler(ABCSampleLabeler):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def label(self, index: pd.DataFrame, health: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        enabled = {str(value) for value in OmegaConf.select(self.cfg, "enabled_for_dataset", default=[])}
        if enabled and not enabled.intersection(set(index["dataset"].astype(str))):
            return pd.DataFrame({"sample_uid": index["sample_uid"]}), {}

        severe_state_id = int(OmegaConf.select(self.cfg, "severe_state_id", default=3))
        normal_label = str(OmegaConf.select(self.cfg, "normal_label", default="normal"))
        degraded_label = str(OmegaConf.select(self.cfg, "degraded_label", default="degraded_unknown"))
        data = index[["sample_uid", "fault_element"]].merge(
            health[["sample_uid", "health_state_id"]],
            on="sample_uid",
            how="left",
        )
        names: List[str] = []
        for _, row in data.iterrows():
            state_id = int(row["health_state_id"])
            if state_id == 0:
                names.append(normal_label)
            elif state_id == severe_state_id:
                names.append(str(row["fault_element"] or degraded_label))
            else:
                names.append(degraded_label)

        mapping = _build_mapping(names, normal_label, degraded_label)
        labels = pd.DataFrame({
            "sample_uid": data["sample_uid"],
            "fault_type_stage_id": [mapping[name] for name in names],
            "fault_type_stage_name": names,
        })
        return labels, mapping


def _build_mapping(names: List[str], normal_label: str, degraded_label: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for label in [normal_label, degraded_label, *names]:
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping

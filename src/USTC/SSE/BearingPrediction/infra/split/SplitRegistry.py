"""
Split registry for Stage 1 CLI integration.
"""

from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.split.CrossConditionSplitter import CrossConditionSplitter
from USTC.SSE.BearingPrediction.infra.split.LeaveOneBearingOutSplitter import LeaveOneBearingOutSplitter
from USTC.SSE.BearingPrediction.infra.split.OfficialPHM2012Splitter import OfficialPHM2012Splitter


def build_splitter(cfg: DictConfig):
    name = str(OmegaConf.select(cfg, "name", default=""))
    if name == "xjtu_leave_one_bearing_out":
        return LeaveOneBearingOutSplitter(cfg)
    if name == "xjtu_cross_condition":
        return CrossConditionSplitter(cfg)
    if name == "phm2012_official":
        return OfficialPHM2012Splitter(cfg)
    raise ValueError(f"Unsupported split name: {name}")

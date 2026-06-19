"""
Labeler registry.
"""

from omegaconf import OmegaConf

from USTC.SSE.BearingPrediction.infra.label.CappedRulLabeler import CappedRulLabeler
from USTC.SSE.BearingPrediction.infra.label.EarlyFaultLabeler import EarlyFaultLabeler
from USTC.SSE.BearingPrediction.infra.label.FaultTypeStageLabeler import FaultTypeStageLabeler
from USTC.SSE.BearingPrediction.infra.label.HealthStateLabeler import HealthStateLabeler
from USTC.SSE.BearingPrediction.infra.label.LinearRulLabeler import LinearRulLabeler
from USTC.SSE.BearingPrediction.infra.label.PiecewiseRulLabeler import PiecewiseRulLabeler


def build_labeler(output_cfg):
    label_type = str(OmegaConf.select(output_cfg, "type", default=""))
    params = OmegaConf.create(OmegaConf.select(output_cfg, "params", default={}))
    if label_type == "linear_rul":
        return LinearRulLabeler(params)
    if label_type == "capped_rul":
        return CappedRulLabeler(params)
    if label_type == "piecewise_rul":
        return PiecewiseRulLabeler(params)
    if label_type == "health_state":
        return HealthStateLabeler(params)
    if label_type == "early_fault":
        return EarlyFaultLabeler(params)
    if label_type == "fault_type_stage":
        return FaultTypeStageLabeler(params)
    raise ValueError(f"Unsupported labeler type: {label_type}")

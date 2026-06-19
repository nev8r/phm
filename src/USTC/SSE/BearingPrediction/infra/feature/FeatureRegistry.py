"""
Feature backend registry.
"""

from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.ManualProcessorFeatureBackend import ManualProcessorFeatureBackend
from USTC.SSE.BearingPrediction.infra.feature.TsfreshFeatureBackend import TsfreshFeatureBackend


def build_feature_backend(cfg: DictConfig):
    backend_type = str(OmegaConf.select(cfg, "type", default=""))
    if backend_type == "manual_processor":
        return ManualProcessorFeatureBackend(cfg)
    if backend_type == "tsfresh":
        return TsfreshFeatureBackend(cfg)
    raise ValueError(f"Unsupported feature backend type: {backend_type}")

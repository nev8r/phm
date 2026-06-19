"""
Fault-type feature separability analyzer.
"""

from omegaconf import DictConfig

from USTC.SSE.BearingPrediction.analysis.HealthStateFeatureAnalyzer import HealthStateFeatureAnalyzer


class FaultTypeFeatureAnalyzer(HealthStateFeatureAnalyzer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

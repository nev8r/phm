"""
Fault-type feature separability analyzer.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from omegaconf import DictConfig

from USTC.SSE.BearingPrediction.analysis.HealthStateFeatureAnalyzer import HealthStateFeatureAnalyzer


class FaultTypeFeatureAnalyzer(HealthStateFeatureAnalyzer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

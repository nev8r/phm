"""
Open-source SOTA experiment package

this package exposes SOTA target and reproduction evidence helpers

created by zyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.experiments.sota.sota_protocol import (
    REPRODUCTION_COLUMNS,
    TARGET_COLUMNS,
    SotaReproductionRecord,
    SotaTargetRecord,
    calculate_gap_percent,
    validate_reproduction_frame,
    validate_target_frame,
)
from USTC.SSE.BearingPrediction.experiments.sota.sota_adapters import ExternalSotaAdapter, default_external_adapters
from USTC.SSE.BearingPrediction.experiments.sota.sota_runner import SotaEvidenceBuilder

__all__ = [
    "ExternalSotaAdapter",
    "REPRODUCTION_COLUMNS",
    "TARGET_COLUMNS",
    "SotaEvidenceBuilder",
    "SotaReproductionRecord",
    "SotaTargetRecord",
    "calculate_gap_percent",
    "default_external_adapters",
    "validate_reproduction_frame",
    "validate_target_frame",
]

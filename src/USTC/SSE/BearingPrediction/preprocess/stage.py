"""
Degradation stage module

this file is for implementing multiple degradation stage partition strategies

created by cyy

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.common.registry import ComponentRegistry

STAGE_STRATEGY_REGISTRY = ComponentRegistry("stage_strategy")


@dataclass
class DegradationStageResult:
    """
    Result of stage partitioning.

    Parameters
    ----------
    health_indicator : np.ndarray
        health indicator values
    stage_labels : np.ndarray
        stage labels
    onset_index : int
        degradation onset index
    severe_index : int
        severe degradation index
    stage_names : list[str]
        stage names
    """

    health_indicator: np.ndarray
    stage_labels: np.ndarray
    onset_index: int
    severe_index: int
    stage_names: list[str]

    def as_frame(self) -> pd.DataFrame:
        """
        export stage results to dataframe

        Returns
        -------
        pd.DataFrame
            stage frame
        """

        return pd.DataFrame(
            {
                "sample_index": np.arange(self.health_indicator.size),
                "health_indicator": self.health_indicator,
                "stage_label": self.stage_labels,
                "stage_name": [self.stage_names[label] for label in self.stage_labels],
            }
        )


class StageStrategy:
    """
    Base class for stage partition strategies.
    """

    stage_names = ["healthy", "degrading", "severe"]

    def fit_predict(self, health_indicator: np.ndarray) -> DegradationStageResult:
        raise NotImplementedError


@STAGE_STRATEGY_REGISTRY.register("three_sigma")
class ThreeSigmaStageStrategy(StageStrategy):
    """
    Partition stages with baseline mean plus n-sigma thresholds.
    """

    def __init__(self, baseline_size: int = 10, sigma_level: float = 3.0) -> None:
        self.baseline_size = baseline_size
        self.sigma_level = sigma_level

    def fit_predict(self, health_indicator: np.ndarray) -> DegradationStageResult:
        baseline_values = health_indicator[: max(2, min(self.baseline_size, health_indicator.size))]
        baseline_mean = float(np.mean(baseline_values))
        baseline_std = float(np.std(baseline_values)) + 1e-8
        onset_threshold = baseline_mean + (self.sigma_level * baseline_std)
        severe_threshold = baseline_mean + ((self.sigma_level * 1.8) * baseline_std)

        onset_candidates = np.where(health_indicator >= onset_threshold)[0]
        severe_candidates = np.where(health_indicator >= severe_threshold)[0]
        onset_index = int(onset_candidates[0]) if onset_candidates.size else max(health_indicator.size // 3, 1)
        severe_index = int(severe_candidates[0]) if severe_candidates.size else max((health_indicator.size * 2) // 3, onset_index + 1)
        stage_labels = np.zeros(health_indicator.size, dtype=int)
        stage_labels[onset_index:severe_index] = 1
        stage_labels[severe_index:] = 2
        return DegradationStageResult(health_indicator, stage_labels, onset_index, severe_index, self.stage_names)


@STAGE_STRATEGY_REGISTRY.register("fpt")
class FPTStageStrategy(StageStrategy):
    """
    Estimate first predictable time with persistence on the health indicator.
    """

    def __init__(self, baseline_size: int = 10, sigma_level: float = 2.5, persistence: int = 3) -> None:
        self.baseline_size = baseline_size
        self.sigma_level = sigma_level
        self.persistence = persistence

    def fit_predict(self, health_indicator: np.ndarray) -> DegradationStageResult:
        baseline_values = health_indicator[: max(2, min(self.baseline_size, health_indicator.size))]
        baseline_mean = float(np.mean(baseline_values))
        baseline_std = float(np.std(baseline_values)) + 1e-8
        threshold = baseline_mean + (self.sigma_level * baseline_std)

        onset_index = max(health_indicator.size // 3, 1)
        for start_index in range(0, max(1, health_indicator.size - self.persistence + 1)):
            window = health_indicator[start_index : start_index + self.persistence]
            if np.all(window >= threshold):
                onset_index = start_index
                break

        severe_index = onset_index + max((health_indicator.size - onset_index) // 2, 1)
        severe_index = min(severe_index, health_indicator.size - 1)
        stage_labels = np.zeros(health_indicator.size, dtype=int)
        stage_labels[onset_index:severe_index] = 1
        stage_labels[severe_index:] = 2
        return DegradationStageResult(health_indicator, stage_labels, onset_index, severe_index, self.stage_names)


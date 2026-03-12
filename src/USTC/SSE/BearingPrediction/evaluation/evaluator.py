"""
Evaluator module

this file is for aggregating multiple evaluation metrics

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import numpy as np

from USTC.SSE.BearingPrediction.evaluation.metrics import Metric


class Evaluator:
    """
    Evaluate predictions with a configurable set of metrics.
    """

    def __init__(self) -> None:
        self.metrics: list[Metric] = []

    def add(self, *metrics: Metric) -> "Evaluator":
        """
        register metrics

        Parameters
        ----------
        metrics : Metric
            metric instances

        Returns
        -------
        Evaluator
            evaluator instance
        """

        self.metrics.extend(metrics)
        return self

    def evaluate(self, targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        """
        compute all configured metrics

        Parameters
        ----------
        targets : np.ndarray
            ground truth
        predictions : np.ndarray
            predictions

        Returns
        -------
        dict[str, float]
            metric mapping
        """

        return {metric.name: metric(np.asarray(targets), np.asarray(predictions)) for metric in self.metrics}

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        return self.evaluate(targets, predictions)


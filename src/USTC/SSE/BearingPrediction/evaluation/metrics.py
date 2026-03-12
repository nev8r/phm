"""
Metric module

this file is for implementing regression and classification evaluation metrics

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metric:
    """
    Base metric class.
    """

    name: str

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        raise NotImplementedError


class MAE(Metric):
    def __init__(self) -> None:
        super().__init__("mae")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(np.abs(predictions - targets)))


class MSE(Metric):
    def __init__(self) -> None:
        super().__init__("mse")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(np.square(predictions - targets)))


class RMSE(Metric):
    def __init__(self) -> None:
        super().__init__("rmse")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(predictions - targets))))


class MAPE(Metric):
    def __init__(self) -> None:
        super().__init__("mape")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        safe_denominator = np.maximum(np.abs(targets), 1.0)
        return float(np.mean(np.abs((predictions - targets) / safe_denominator)))


class PercentError(Metric):
    def __init__(self) -> None:
        super().__init__("percent_error")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        safe_denominator = np.maximum(np.abs(targets), 1.0)
        return float(np.mean(((predictions - targets) / safe_denominator) * 100.0))


class PHM2012Score(Metric):
    """
    Asymmetric challenge style score for bearing RUL prediction.
    """

    def __init__(self) -> None:
        super().__init__("phm2012_score")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        diff = predictions - targets
        score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
        return float(np.sum(score))


class PHM2008Score(Metric):
    """
    NASA style asymmetric score used by classical prognostics benchmarks.
    """

    def __init__(self) -> None:
        super().__init__("phm2008_score")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        diff = predictions - targets
        score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
        return float(np.sum(score))


class NASAScore(Metric):
    def __init__(self) -> None:
        super().__init__("nasa_score")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        diff = predictions - targets
        return float(np.sum(np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)))


class Accuracy(Metric):
    def __init__(self) -> None:
        super().__init__("accuracy")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(np.asarray(targets).astype(int) == np.asarray(predictions).astype(int)))

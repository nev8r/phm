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


class NormalizedRMSE(Metric):
    """
    Range-normalized root mean square error for RUL prediction.
    """

    def __init__(self, normalization: str = "range") -> None:
        super().__init__("normalized_rmse")
        self.normalization = normalization

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        rmse_value = RMSE()(targets, predictions)
        if self.normalization != "range":
            raise ValueError("NormalizedRMSE currently supports only range normalization")
        denominator = float(np.max(targets) - np.min(targets)) if targets.size else 0.0
        if abs(denominator) < 1e-8:
            denominator = 1.0
        return float(rmse_value / denominator)


class MAPE(Metric):
    def __init__(self) -> None:
        super().__init__("mape")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        safe_denominator = np.maximum(np.abs(targets), 1.0)
        return float(np.mean(np.abs((predictions - targets) / safe_denominator)))


class SMAPE(Metric):
    """
    Symmetric mean absolute percentage error.
    """

    def __init__(self) -> None:
        super().__init__("smape")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        denominator = np.abs(targets) + np.abs(predictions)
        values = np.divide(
            2.0 * np.abs(predictions - targets),
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 1e-8,
        )
        return float(np.mean(values))


class PercentError(Metric):
    def __init__(self) -> None:
        super().__init__("percent_error")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        safe_denominator = np.maximum(np.abs(targets), 1.0)
        return float(np.mean(((predictions - targets) / safe_denominator) * 100.0))


class R2Score(Metric):
    def __init__(self) -> None:
        super().__init__("r2")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        residual_sum = float(np.sum(np.square(targets - predictions)))
        total_sum = float(np.sum(np.square(targets - np.mean(targets)))) if targets.size else 0.0
        if total_sum < 1e-8:
            return 1.0 if residual_sum < 1e-8 else 0.0
        return float(1.0 - (residual_sum / total_sum))


class MedianAbsoluteError(Metric):
    def __init__(self) -> None:
        super().__init__("median_absolute_error")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.median(np.abs(predictions - targets)))


class MaxAbsoluteError(Metric):
    def __init__(self) -> None:
        super().__init__("max_absolute_error")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.max(np.abs(predictions - targets))) if targets.size else 0.0


class MeanError(Metric):
    def __init__(self) -> None:
        super().__init__("mean_error")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(predictions - targets))


class OverPredictionRate(Metric):
    def __init__(self) -> None:
        super().__init__("over_prediction_rate")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(predictions > targets))


class UnderPredictionRate(Metric):
    def __init__(self) -> None:
        super().__init__("under_prediction_rate")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(predictions < targets))


class WithinToleranceRate(Metric):
    """
    Fraction of predictions within an absolute or relative error tolerance.
    """

    def __init__(self, tolerance: float = 0.10, *, relative: bool = True) -> None:
        self.tolerance = tolerance
        self.relative = relative
        metric_name = f"within_{int(round(tolerance * 100))}_percent_rate" if relative else f"within_{tolerance:g}_unit_rate"
        super().__init__(metric_name)

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        absolute_errors = np.abs(predictions - targets)
        if self.relative:
            denominator = np.maximum(np.abs(targets), 1.0)
            return float(np.mean((absolute_errors / denominator) <= self.tolerance))
        return float(np.mean(absolute_errors <= self.tolerance))


class HuangRulScore(Metric):
    """
    Score function from the CNN-LSTM-AM RUL paper by Huang et al.
    """

    def __init__(self) -> None:
        super().__init__("huang_rul_score")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        safe_denominator = np.maximum(np.abs(targets), 1.0)
        percentage_errors = 100.0 * ((targets - predictions) / safe_denominator)
        positive_factor = -np.log(0.5)
        scores = np.where(
            percentage_errors <= 0.0,
            np.exp(positive_factor * (percentage_errors / 5.0)),
            np.exp(positive_factor * (percentage_errors / 20.0)),
        )
        return float(np.mean(scores))


class AsymmetricRulPenalty(Metric):
    """
    Configurable exponential penalty for early and late RUL prediction errors.
    """

    def __init__(
        self,
        *,
        under_prediction_scale: float = 13.0,
        over_prediction_scale: float = 10.0,
        name: str = "asymmetric_rul_penalty",
    ) -> None:
        super().__init__(name)
        self.under_prediction_scale = under_prediction_scale
        self.over_prediction_scale = over_prediction_scale

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        diff = predictions - targets
        penalty = np.where(
            diff < 0.0,
            np.exp(-diff / self.under_prediction_scale) - 1.0,
            np.exp(diff / self.over_prediction_scale) - 1.0,
        )
        return float(np.sum(penalty))


class PHM2012Score(AsymmetricRulPenalty):
    """
    Asymmetric challenge style score for bearing RUL prediction.
    """

    def __init__(self) -> None:
        super().__init__(under_prediction_scale=13.0, over_prediction_scale=10.0, name="phm2012_score")


class PHM2008Score(AsymmetricRulPenalty):
    """
    NASA style asymmetric score used by classical prognostics benchmarks.
    """

    def __init__(self) -> None:
        super().__init__(under_prediction_scale=13.0, over_prediction_scale=10.0, name="phm2008_score")


class NASAScore(AsymmetricRulPenalty):
    def __init__(self) -> None:
        super().__init__(under_prediction_scale=13.0, over_prediction_scale=10.0, name="nasa_score")


class Accuracy(Metric):
    def __init__(self) -> None:
        super().__init__("accuracy")

    def __call__(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(np.asarray(targets).astype(int) == np.asarray(predictions).astype(int)))

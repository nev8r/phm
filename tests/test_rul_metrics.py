"""
RUL metric tests

this file is for testing RUL-oriented regression and paper score metrics

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import math

import numpy as np

from USTC.SSE.BearingPrediction.api import (
    AsymmetricRulPenalty,
    HuangRulScore,
    MaxAbsoluteError,
    MeanError,
    MedianAbsoluteError,
    NormalizedRMSE,
    OverPredictionRate,
    R2Score,
    SMAPE,
    UnderPredictionRate,
    WithinToleranceRate,
)


def test_huang_rul_score_matches_paper_piecewise_formula() -> None:
    targets = np.asarray([100.0, 100.0, 100.0])
    predictions = np.asarray([100.0, 110.0, 80.0])

    score = HuangRulScore()(targets, predictions)

    expected = (
        1.0
        + math.exp(-math.log(0.5) * (-10.0 / 5.0))
        + math.exp(-math.log(0.5) * (20.0 / 20.0))
    ) / 3.0
    assert score == expected


def test_common_rul_regression_metrics_have_expected_values() -> None:
    targets = np.asarray([10.0, 20.0, 30.0])
    predictions = np.asarray([10.0, 25.0, 20.0])

    assert NormalizedRMSE()(targets, predictions) == math.sqrt((0.0 + 25.0 + 100.0) / 3.0) / 20.0
    assert SMAPE()(targets, predictions) == np.mean([0.0, 2.0 * 5.0 / 45.0, 2.0 * 10.0 / 50.0])
    assert R2Score()(targets, predictions) == 1.0 - (125.0 / 200.0)
    assert MedianAbsoluteError()(targets, predictions) == 5.0
    assert MaxAbsoluteError()(targets, predictions) == 10.0
    assert MeanError()(targets, predictions) == np.mean([0.0, 5.0, -10.0])


def test_perfect_prediction_metrics_are_zero_or_one() -> None:
    targets = np.asarray([4.0, 8.0, 12.0])
    predictions = targets.copy()

    assert NormalizedRMSE()(targets, predictions) == 0.0
    assert SMAPE()(targets, predictions) == 0.0
    assert MedianAbsoluteError()(targets, predictions) == 0.0
    assert MaxAbsoluteError()(targets, predictions) == 0.0
    assert MeanError()(targets, predictions) == 0.0
    assert R2Score()(targets, predictions) == 1.0


def test_directional_and_tolerance_metrics_report_prediction_bias() -> None:
    targets = np.asarray([100.0, 100.0, 100.0, 100.0])
    predictions = np.asarray([110.0, 80.0, 95.0, 100.0])

    assert OverPredictionRate()(targets, predictions) == 0.25
    assert UnderPredictionRate()(targets, predictions) == 0.5
    assert WithinToleranceRate(tolerance=0.10)(targets, predictions) == 0.75
    assert WithinToleranceRate(tolerance=5.0, relative=False)(targets, predictions) == 0.5


def test_normalized_rmse_handles_zero_range_targets() -> None:
    targets = np.asarray([5.0, 5.0, 5.0])
    predictions = np.asarray([5.0, 6.0, 4.0])

    value = NormalizedRMSE()(targets, predictions)

    assert np.isfinite(value)
    assert value == math.sqrt(2.0 / 3.0)


def test_asymmetric_rul_penalty_uses_configurable_early_and_late_scales() -> None:
    targets = np.asarray([100.0, 100.0])
    predictions = np.asarray([90.0, 120.0])

    penalty = AsymmetricRulPenalty(under_prediction_scale=10.0, over_prediction_scale=20.0)

    expected = (math.exp(10.0 / 10.0) - 1.0) + (math.exp(20.0 / 20.0) - 1.0)
    assert penalty(targets, predictions) == expected

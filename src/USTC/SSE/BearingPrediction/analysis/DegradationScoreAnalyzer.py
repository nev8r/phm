"""
Degradation behavior score analyzer.
"""

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.analysis._helpers import feature_columns, safe_corr, safe_numeric


class DegradationScoreAnalyzer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def analyze(self, features: pd.DataFrame) -> pd.DataFrame:
        interpolation_points = int(OmegaConf.select(self.cfg, "interpolation_points", default=100))
        rows: List[Dict] = []
        for column in feature_columns(features):
            rows.append({
                "feature": column,
                "monotonicity": _monotonicity(features, column),
                "robustness": _robustness(features, column),
                "trendability": _trendability(features, column, interpolation_points),
                "prognosability": _prognosability(features, column),
            })
        return pd.DataFrame(rows)


def _monotonicity(features: pd.DataFrame, column: str) -> float:
    scores = []
    for _, group in features.groupby(["dataset", "bearing_id"], sort=False):
        values = safe_numeric(group.sort_values("timestep")[column]).dropna().to_numpy()
        if len(values) < 2:
            continue
        diff = np.diff(values)
        positive = int((diff > 0).sum())
        negative = int((diff < 0).sum())
        scores.append(abs(positive - negative) / max(len(diff), 1))
    return float(np.mean(scores)) if scores else 0.0


def _robustness(features: pd.DataFrame, column: str) -> float:
    scores = []
    for _, group in features.groupby(["dataset", "bearing_id"], sort=False):
        values = safe_numeric(group.sort_values("timestep")[column]).dropna()
        if len(values) < 2:
            continue
        smooth = values.rolling(window=3, min_periods=1).mean()
        residual = values - smooth
        scores.append(1.0 / (1.0 + float(residual.std(ddof=0)) / (float(values.std(ddof=0)) + 1.0e-12)))
    return float(np.mean(scores)) if scores else 0.0


def _trendability(features: pd.DataFrame, column: str, interpolation_points: int) -> float:
    curves = []
    grid = np.linspace(0, 1, interpolation_points)
    for _, group in features.groupby(["dataset", "bearing_id"], sort=False):
        values = safe_numeric(group.sort_values("timestep")[column]).dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        x = np.linspace(0, 1, len(values))
        curves.append(np.interp(grid, x, values))
    if len(curves) < 2:
        return 0.0
    correlations = []
    for left, right in combinations(curves, 2):
        correlations.append(abs(safe_corr(pd.Series(left), pd.Series(right), "pearson")))
    return float(np.mean(correlations)) if correlations else 0.0


def _prognosability(features: pd.DataFrame, column: str) -> float:
    end_values = []
    ranges = []
    for _, group in features.groupby(["dataset", "bearing_id"], sort=False):
        values = safe_numeric(group.sort_values("timestep")[column]).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        end_values.append(values[-1])
        ranges.append(float(np.max(values) - np.min(values)))
    if not end_values:
        return 0.0
    return float(np.exp(-np.std(end_values) / (np.mean(ranges) + 1.0e-12)))

"""
Three-sigma FPT detector.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.degradation.FPTResult import FPTResult


class ThreeSigmaFPTDetector:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def detect(self, hi: pd.DataFrame, source_column: str) -> Dict:
        if "hi_smooth" not in hi.columns:
            raise ValueError("hi must contain hi_smooth")
        results: List[Dict] = []
        for _, group in hi.groupby(["dataset", "bearing_id"], sort=False):
            results.append(self._detect_for_bearing(group, source_column).to_dict())
        return {
            "method": str(OmegaConf.select(self.cfg, "method", default="three_sigma")),
            "source": source_column,
            "results": results,
        }

    def _detect_for_bearing(self, group: pd.DataFrame, source_column: str) -> FPTResult:
        del source_column
        group = group.sort_values("timestep").reset_index(drop=True)
        hi_values = group["hi_smooth"].to_numpy(dtype=float)
        n = len(hi_values)
        if n == 0:
            raise ValueError("Cannot detect FPT for an empty bearing group")

        healthy_ratio = float(OmegaConf.select(self.cfg, "healthy_ratio", default=0.2))
        sigma_ratio = float(OmegaConf.select(self.cfg, "sigma_ratio", default=3.0))
        min_delta = float(OmegaConf.select(self.cfg, "min_delta", default=0.0))
        consecutive_points = max(1, int(OmegaConf.select(self.cfg, "consecutive_points", default=3)))
        healthy_n = _clamp(int(n * healthy_ratio), 1, n)
        baseline = hi_values[:healthy_n]
        threshold = float(baseline.mean() + sigma_ratio * baseline.std(ddof=0) + min_delta)

        fpt_index, success, fallback_used = self._find_crossing(hi_values, threshold, consecutive_points)
        if fpt_index is None:
            fpt_index = self._fallback_index(n, healthy_n)
            success = False
            fallback_used = True

        fpt_index = _clamp(int(fpt_index), 0, n - 1)
        fpt_row = group.iloc[fpt_index]
        return FPTResult(
            dataset=str(fpt_row["dataset"]),
            bearing_id=str(fpt_row["bearing_id"]),
            condition_id=str(fpt_row["condition_id"]),
            fpt_index=fpt_index,
            fpt_sample_uid=str(fpt_row["sample_uid"]),
            fpt_timestep=int(fpt_row["timestep"]),
            threshold=threshold,
            method=str(OmegaConf.select(self.cfg, "method", default="three_sigma")),
            success=bool(success),
            fallback_used=bool(fallback_used),
            params=OmegaConf.to_container(self.cfg, resolve=True),
        )

    @staticmethod
    def _find_crossing(hi_values: np.ndarray, threshold: float, consecutive_points: int):
        n = len(hi_values)
        if consecutive_points > n:
            return None, False, False
        for index in range(0, n - consecutive_points + 1):
            window = hi_values[index:index + consecutive_points]
            if np.all(window > threshold):
                return index, True, False
        return None, False, False

    def _fallback_index(self, n: int, healthy_n: int) -> int:
        fallback = str(OmegaConf.select(self.cfg, "fallback", default="healthy_ratio"))
        if fallback == "healthy_ratio":
            return _clamp(healthy_n, 0, n - 1)
        if fallback == "zero":
            return 0
        if fallback == "last":
            return n - 1
        raise ValueError(f"Unsupported FPT fallback: {fallback}")


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))

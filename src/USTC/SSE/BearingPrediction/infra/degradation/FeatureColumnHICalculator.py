"""
Feature-column health indicator calculator.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.degradation.HealthIndicatorCalculator import HealthIndicatorCalculator
from USTC.SSE.BearingPrediction.infra.degradation.HealthIndicatorFrame import HI_INDEX_COLUMNS, HealthIndicatorFrame
from USTC.SSE.BearingPrediction.infra.degradation.HealthIndicatorSpec import HealthIndicatorSpec


class FeatureColumnHICalculator(HealthIndicatorCalculator):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def calculate(self, features: pd.DataFrame) -> HealthIndicatorFrame:
        source_column = self._select_source_column(features)
        data = features[list(HI_INDEX_COLUMNS)].copy()
        direction = str(OmegaConf.select(self.cfg, "direction", default="bad_high"))
        hi_raw = features[source_column].astype(float)
        if direction == "bad_low":
            hi_raw = -hi_raw
        elif direction != "bad_high":
            raise ValueError(f"Unsupported HI direction: {direction}")

        data["hi_raw"] = hi_raw.to_numpy()
        data = data.sort_values(["dataset", "bearing_id", "timestep"]).reset_index(drop=True)
        data["hi_smooth"] = self._smooth(data)
        if bool(OmegaConf.select(self.cfg, "normalize_per_bearing", default=True)):
            data["hi_norm"] = data.groupby(["dataset", "bearing_id"], group_keys=False)["hi_smooth"].transform(_minmax)
        else:
            data["hi_norm"] = _minmax(data["hi_smooth"])
        data["hi_source_column"] = source_column

        spec = HealthIndicatorSpec(
            method="feature_column",
            params=OmegaConf.to_container(self.cfg, resolve=True),
        ).to_dict()
        frame = HealthIndicatorFrame(
            data=data,
            index_columns=list(HI_INDEX_COLUMNS),
            hi_column="hi_norm",
            spec=spec,
        )
        frame.validate()
        return frame

    def _select_source_column(self, features: pd.DataFrame) -> str:
        candidates = list(OmegaConf.select(self.cfg, "source_column_candidates", default=[]))
        for column in candidates:
            if column in features.columns:
                return column
        raise ValueError(f"None of HI source_column_candidates exist in features: {candidates}")

    def _smooth(self, data: pd.DataFrame) -> pd.Series:
        smooth_cfg = OmegaConf.select(self.cfg, "smooth", default={})
        if not bool(OmegaConf.select(smooth_cfg, "enabled", default=False)):
            return data["hi_raw"]
        window = int(OmegaConf.select(smooth_cfg, "window", default=3))
        return data.groupby(["dataset", "bearing_id"], group_keys=False)["hi_raw"].transform(
            lambda series: series.rolling(window=window, min_periods=1).mean()
        )


def _minmax(series: pd.Series) -> pd.Series:
    min_value = float(series.min())
    max_value = float(series.max())
    denominator = max_value - min_value
    if denominator <= np.finfo(float).eps:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / denominator

"""
Feature cleaner.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS


class FeatureCleaner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.enabled = bool(OmegaConf.select(cfg, "enabled", default=True))
        self.imputer = str(OmegaConf.select(cfg, "imputer", default="median"))
        self.scaler = str(OmegaConf.select(cfg, "scaler", default="standard"))
        self.drop_constant = bool(OmegaConf.select(cfg, "drop_constant", default=True))
        self.constant_threshold = float(OmegaConf.select(cfg, "constant_threshold", default=1.0e-12))
        self.feature_columns: List[str] = []
        self.dropped_columns: List[str] = []
        self.imputer_values: Dict[str, float] = {}
        self.scaler_mean: Dict[str, float] = {}
        self.scaler_std: Dict[str, float] = {}

    def fit(self, features: pd.DataFrame, train_sample_uids: Optional[List[str]] = None) -> "FeatureCleaner":
        self.feature_columns = [column for column in features.columns if column not in FEATURE_INDEX_COLUMNS]
        fit_data = self._fit_subset(features, train_sample_uids)
        fit_values = fit_data[self.feature_columns].replace([np.inf, -np.inf], np.nan)

        self.imputer_values = {}
        for column in self.feature_columns:
            series = fit_values[column]
            value = series.median() if self.imputer == "median" else 0.0
            self.imputer_values[column] = 0.0 if pd.isna(value) else float(value)

        imputed_fit = fit_values.fillna(self.imputer_values)
        self.dropped_columns = []
        if self.drop_constant:
            for column in self.feature_columns:
                if float(imputed_fit[column].std(ddof=0)) <= self.constant_threshold:
                    self.dropped_columns.append(column)

        kept = [column for column in self.feature_columns if column not in set(self.dropped_columns)]
        self.scaler_mean = {}
        self.scaler_std = {}
        if self.scaler == "standard":
            for column in kept:
                mean = float(imputed_fit[column].mean())
                std = float(imputed_fit[column].std(ddof=0))
                self.scaler_mean[column] = mean
                self.scaler_std[column] = std if std > self.constant_threshold else 1.0
        else:
            for column in kept:
                self.scaler_mean[column] = 0.0
                self.scaler_std[column] = 1.0

        self.feature_columns = kept
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return features.copy()

        cleaned = features[FEATURE_INDEX_COLUMNS].copy()
        values = features[[*self.feature_columns]].replace([np.inf, -np.inf], np.nan)
        fill_values = {column: self.imputer_values[column] for column in self.feature_columns}
        values = values.fillna(fill_values)
        for column in self.feature_columns:
            values[column] = (values[column] - self.scaler_mean[column]) / self.scaler_std[column]
        cleaned = pd.concat([cleaned, values], axis=1)
        return cleaned

    def fit_transform(self, features: pd.DataFrame, train_sample_uids: Optional[List[str]] = None) -> pd.DataFrame:
        self.fit(features, train_sample_uids)
        return self.transform(features)

    def state_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "dropped_columns": self.dropped_columns,
            "imputer_values": self.imputer_values,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }

    def _fit_subset(self, features: pd.DataFrame, train_sample_uids: Optional[List[str]]) -> pd.DataFrame:
        if train_sample_uids is None:
            return features
        subset = features[features["sample_uid"].isin(train_sample_uids)]
        if subset.empty:
            raise ValueError("FeatureCleaner train_sample_uids produced an empty fit subset")
        return subset

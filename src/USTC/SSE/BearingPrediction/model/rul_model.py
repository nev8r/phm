"""
RUL prediction model module

this file is for training and evaluating remaining useful life models

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from USTC.SSE.BearingPrediction.config import TrainingConfig

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled at runtime when dependency is unavailable
    XGBRegressor = None


@dataclass(frozen=True)
class RULModelResult:
    """
    Container for model evaluation results.

    Parameters
    ----------
    metrics : dict[str, float]
        metric values
    predictions : pd.DataFrame
        test prediction results
    """

    metrics: dict[str, float]
    predictions: pd.DataFrame


class RULPredictionModel:
    """
    Train an xgboost based regressor for RUL prediction.
    """

    identifier_columns = {"bearing_id", "cycle", "window_index", "failure_cycle", "rul", "duration", "event"}

    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config
        self.feature_columns: list[str] = []
        self.estimator = self._build_estimator()

    def train_and_evaluate(self, feature_table: pd.DataFrame) -> RULModelResult:
        """
        split data, train model, and evaluate prediction performance

        Parameters
        ----------
        feature_table : pd.DataFrame
            model feature table

        Returns
        -------
        RULModelResult
            evaluation result
        """

        self.feature_columns = [
            column_name for column_name in feature_table.columns if column_name not in self.identifier_columns
        ]

        train_frame, test_frame = self._split_feature_table(feature_table)

        train_features = train_frame[self.feature_columns]
        test_features = test_frame[self.feature_columns]
        train_target = train_frame["rul"]
        test_target = test_frame["rul"]

        self.estimator.fit(train_features, train_target)
        predicted_rul = self.estimator.predict(test_features)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(test_target, predicted_rul))),
            "mae": float(mean_absolute_error(test_target, predicted_rul)),
        }
        predictions = test_frame[["bearing_id", "cycle", "window_index", "rul"]].copy()
        predictions["predicted_rul"] = predicted_rul
        predictions = predictions.sort_values("rul", ascending=False).reset_index(drop=True)
        return RULModelResult(metrics=metrics, predictions=predictions)

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """
        predict rul for new observations

        Parameters
        ----------
        feature_frame : pd.DataFrame
            feature input frame

        Returns
        -------
        np.ndarray
            predicted rul values
        """

        return self.estimator.predict(feature_frame[self.feature_columns])

    def _split_feature_table(self, feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        split feature table by bearing id to avoid leakage across windows

        Parameters
        ----------
        feature_table : pd.DataFrame
            feature table

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            train and test frame
        """

        unique_bearings = feature_table["bearing_id"].nunique()
        if unique_bearings >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.training_config.test_size,
                random_state=self.training_config.random_state,
            )
            train_index, test_index = next(splitter.split(feature_table, groups=feature_table["bearing_id"]))
            train_frame = feature_table.iloc[train_index].copy()
            test_frame = feature_table.iloc[test_index].copy()
            return train_frame, test_frame

        train_frame, test_frame = train_test_split(
            feature_table,
            test_size=self.training_config.test_size,
            random_state=self.training_config.random_state,
            shuffle=True,
        )
        return train_frame.copy(), test_frame.copy()

    def _build_estimator(self):
        """
        build model estimator with xgboost fallback

        Returns
        -------
        object
            estimator instance
        """

        if XGBRegressor is not None:
            return XGBRegressor(
                objective="reg:squarederror",
                n_estimators=240,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.training_config.random_state,
            )

        return GradientBoostingRegressor(random_state=self.training_config.random_state)

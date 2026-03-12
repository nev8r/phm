"""
Survival model module

this file is for training and evaluating survival analysis models

created by zdh

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from USTC.SSE.BearingPrediction.config import TrainingConfig


@dataclass(frozen=True)
class SurvivalAnalysisResult:
    """
    Container for survival analysis outputs.

    Parameters
    ----------
    metrics : dict[str, float]
        survival evaluation metrics
    survival_curve : pd.DataFrame
        survival curve values
    failure_probability : float
        predicted failure probability at horizon
    """

    metrics: dict[str, float]
    survival_curve: pd.DataFrame
    failure_probability: float


class SurvivalAnalysisService:
    """
    Fit Kaplan-Meier and Cox proportional hazards models.
    """

    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config
        self.kaplan_meier_model = KaplanMeierFitter()
        self.cox_model = CoxPHFitter(penalizer=0.1)
        self.survival_feature_columns = [
            "temperature",
            "load",
            "health_index",
            "rms",
            "kurtosis",
            "crest_factor",
            "dominant_frequency",
            "spectrum_energy",
            "spectral_entropy",
        ]

    def fit_and_evaluate(self, feature_table: pd.DataFrame) -> SurvivalAnalysisResult:
        """
        fit survival models and evaluate predictive quality

        Parameters
        ----------
        feature_table : pd.DataFrame
            model feature table

        Returns
        -------
        SurvivalAnalysisResult
            evaluation result
        """

        survival_frame = feature_table[["bearing_id"] + self.survival_feature_columns + ["duration", "event"]].copy()
        survival_frame = survival_frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        train_frame, test_frame = self._split_survival_frame(survival_frame)
        train_model_frame = train_frame[self.survival_feature_columns + ["duration", "event"]]
        test_model_frame = test_frame[self.survival_feature_columns + ["duration", "event"]]

        self.kaplan_meier_model.fit(durations=train_model_frame["duration"], event_observed=train_model_frame["event"])
        self.cox_model.fit(
            train_model_frame,
            duration_col="duration",
            event_col="event",
            show_progress=False,
        )

        partial_hazard = self.cox_model.predict_partial_hazard(test_model_frame[self.survival_feature_columns]).to_numpy().ravel()
        c_index = float(
            concordance_index(
                event_times=test_model_frame["duration"],
                predicted_scores=-partial_hazard,
                event_observed=test_model_frame["event"],
            )
        )

        prediction_horizon = self.training_config.prediction_horizon
        survival_probabilities = self.predict_survival_probability(
            test_model_frame[self.survival_feature_columns],
            horizon=prediction_horizon,
        )
        observed_survival = (test_model_frame["duration"] > prediction_horizon).astype(float).to_numpy()
        brier_score = float(np.mean(np.square(observed_survival - survival_probabilities)))

        representative_observation = test_model_frame.iloc[[0]][self.survival_feature_columns]
        timeline = np.arange(1, prediction_horizon + 1)
        survival_curve = self.predict_survival_curve(representative_observation, timeline)
        failure_probability = float(1.0 - survival_curve["cox_survival_probability"].iloc[-1])

        return SurvivalAnalysisResult(
            metrics={"c_index": c_index, "brier_score": brier_score},
            survival_curve=survival_curve,
            failure_probability=failure_probability,
        )

    def predict_survival_probability(self, observation_frame: pd.DataFrame, horizon: int) -> np.ndarray:
        """
        predict survival probability for a given horizon

        Parameters
        ----------
        observation_frame : pd.DataFrame
            covariate frame
        horizon : int
            prediction horizon

        Returns
        -------
        np.ndarray
            survival probabilities
        """

        survival_function = self.cox_model.predict_survival_function(observation_frame, times=[horizon])
        return survival_function.iloc[0].to_numpy()

    def predict_survival_curve(self, observation_frame: pd.DataFrame, timeline: np.ndarray) -> pd.DataFrame:
        """
        predict survival curve for a single observation

        Parameters
        ----------
        observation_frame : pd.DataFrame
            covariate frame
        timeline : np.ndarray
            timeline values

        Returns
        -------
        pd.DataFrame
            survival curve frame
        """

        km_curve = self.kaplan_meier_model.predict(timeline).to_numpy()
        cox_curve = self.cox_model.predict_survival_function(observation_frame, times=timeline).iloc[:, 0].to_numpy()
        return pd.DataFrame(
            {
                "timeline": timeline,
                "kaplan_meier_survival_probability": km_curve,
                "cox_survival_probability": cox_curve,
            }
        )

    def _split_survival_frame(self, survival_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        split survival frame by bearing id for unbiased evaluation

        Parameters
        ----------
        survival_frame : pd.DataFrame
            survival feature table

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            train and test frame
        """

        unique_bearings = survival_frame["bearing_id"].nunique()
        if unique_bearings >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.training_config.test_size,
                random_state=self.training_config.random_state,
            )
            train_index, test_index = next(splitter.split(survival_frame, groups=survival_frame["bearing_id"]))
            return survival_frame.iloc[train_index].copy(), survival_frame.iloc[test_index].copy()

        train_frame, test_frame = train_test_split(
            survival_frame,
            test_size=self.training_config.test_size,
            random_state=self.training_config.random_state,
            shuffle=True,
        )
        return train_frame.copy(), test_frame.copy()

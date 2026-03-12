"""
Tester module

this file is for model inference and result collection

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from USTC.SSE.BearingPrediction.prediction.strategies import DirectPredictor


@dataclass
class TestResult:
    """
    Inference result container.

    Parameters
    ----------
    predictions : np.ndarray
        model predictions
    targets : np.ndarray
        ground truth values
    metadata_frame : pd.DataFrame
        metadata frame
    uncertainties : np.ndarray | None
        uncertainty values
    attention_weights : np.ndarray | None
        attention map
    """

    predictions: np.ndarray
    targets: np.ndarray
    metadata_frame: pd.DataFrame
    uncertainties: np.ndarray | None = None
    attention_weights: np.ndarray | None = None

    def as_frame(self) -> pd.DataFrame:
        """
        convert inference outputs to dataframe

        Returns
        -------
        pd.DataFrame
            prediction dataframe
        """

        frame = self.metadata_frame.copy()
        frame["target"] = self.targets
        frame["prediction"] = self.predictions
        if self.uncertainties is not None:
            frame["uncertainty"] = self.uncertainties
        return frame


class BaseTester:
    """
    Evaluate a trained model on a dataset.
    """

    def __init__(self, *, device: str | None = None, batch_size: int = 64) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size

    def test(self, model: torch.nn.Module, dataset: Any, predictor: Any | None = None) -> TestResult:
        """
        run inference on a dataset

        Parameters
        ----------
        model : torch.nn.Module
            trained model
        dataset : Any
            input dataset
        predictor : Any | None
            predictor strategy

        Returns
        -------
        TestResult
            test result
        """

        predictor = predictor or DirectPredictor()
        prediction_bundle = predictor.predict(model, dataset, device=self.device, batch_size=self.batch_size)
        attention_weights = model.maybe_get_attention() if hasattr(model, "maybe_get_attention") else None
        return TestResult(
            predictions=prediction_bundle["predictions"],
            targets=prediction_bundle["targets"],
            metadata_frame=dataset.metadata_frame.copy(),
            uncertainties=prediction_bundle.get("uncertainties"),
            attention_weights=attention_weights.detach().cpu().numpy() if attention_weights is not None else None,
        )


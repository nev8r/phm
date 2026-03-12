"""
Prediction strategy module

this file is for implementing direct, rolling and uncertainty aware prediction

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


class DirectPredictor:
    """
    Standard one-pass prediction strategy.
    """

    def predict(self, model: torch.nn.Module, dataset: Any, *, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
        model = model.to(device)
        model.eval()
        predictions: list[float] = []
        targets: list[float] = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["features"].to(device) if getattr(model, "input_kind", "sequence") == "feature" and "features" in batch else batch["inputs"].to(device)
                outputs = model(inputs)["prediction"]
                if dataset.task_type == "classification":
                    batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    batch_predictions = outputs.squeeze(-1).cpu().numpy()
                predictions.extend(batch_predictions.tolist())
                targets.extend(batch["targets"].cpu().numpy().reshape(-1).tolist())

        return {
            "predictions": np.asarray(predictions),
            "targets": np.asarray(targets),
        }


class MonteCarloDropoutPredictor(DirectPredictor):
    """
    Predict mean and uncertainty via Monte Carlo dropout.
    """

    def __init__(self, passes: int = 10) -> None:
        self.passes = passes

    def predict(self, model: torch.nn.Module, dataset: Any, *, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
        model = model.to(device)
        model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_pass_predictions: list[np.ndarray] = []
        targets: np.ndarray | None = None

        for _ in range(self.passes):
            pass_predictions: list[np.ndarray] = []
            pass_targets: list[np.ndarray] = []
            for batch in loader:
                inputs = batch["features"].to(device) if getattr(model, "input_kind", "sequence") == "feature" and "features" in batch else batch["inputs"].to(device)
                outputs = model(inputs)["prediction"]
                pass_predictions.append(outputs.squeeze(-1).detach().cpu().numpy())
                pass_targets.append(batch["targets"].cpu().numpy().reshape(-1))
            all_pass_predictions.append(np.concatenate(pass_predictions, axis=0))
            if targets is None:
                targets = np.concatenate(pass_targets, axis=0)

        prediction_stack = np.stack(all_pass_predictions, axis=0)
        return {
            "predictions": prediction_stack.mean(axis=0),
            "targets": targets if targets is not None else np.array([]),
            "uncertainties": prediction_stack.std(axis=0),
        }


class RollingPredictor:
    """
    Recursive multi-step predictor for scalar sequence forecasting datasets.
    """

    def predict_sequence(
        self,
        model: torch.nn.Module,
        history: np.ndarray,
        *,
        steps: int,
        device: torch.device,
    ) -> np.ndarray:
        """
        roll forward a scalar history sequence

        Parameters
        ----------
        model : torch.nn.Module
            forecasting model
        history : np.ndarray
            historical window
        steps : int
            number of recursive steps
        device : torch.device
            device

        Returns
        -------
        np.ndarray
            future predictions
        """

        current_window = np.asarray(history, dtype=np.float32).copy()
        forecasts: list[float] = []
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(steps):
                input_tensor = torch.tensor(current_window[None, :], dtype=torch.float32, device=device)
                next_value = float(model(input_tensor)["prediction"].squeeze().cpu())
                forecasts.append(next_value)
                current_window = np.concatenate([current_window[1:], np.asarray([next_value], dtype=np.float32)])
        return np.asarray(forecasts, dtype=np.float32)


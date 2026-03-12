"""
Trainer module

this file is for model training with callbacks and experiment recording

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from USTC.SSE.BearingPrediction.training.callbacks import Callback
from USTC.SSE.BearingPrediction.training.experiment import ExperimentConfig, ExperimentTracker


@dataclass
class TrainingResult:
    """
    Training result container.

    Parameters
    ----------
    best_epoch : int
        best validation epoch
    best_metric : float
        best monitored metric
    history : list[dict[str, float]]
        metric history
    """

    best_epoch: int
    best_metric: float
    history: list[dict[str, float]]


class BaseTrainer:
    """
    Generic trainer for regression and classification bearing models.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        callbacks: list[Callback] | None = None,
        experiment_tracker: ExperimentTracker | None = None,
        batch_size: int = 32,
        max_epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_workers: int = 0,
        gradient_clip_norm: float = 5.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.callbacks = callbacks or []
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.gradient_clip_norm = gradient_clip_norm
        self.experiment_tracker = experiment_tracker
        self.model: nn.Module | None = None
        self.should_stop = False
        self.global_step = 0
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        self.best_metric = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def train(
        self,
        model: nn.Module,
        train_set: Any,
        valid_set: Any | None = None,
    ) -> TrainingResult:
        """
        train a model with optional validation set

        Parameters
        ----------
        model : nn.Module
            model to train
        train_set : Any
            training dataset
        valid_set : Any | None
            validation dataset

        Returns
        -------
        TrainingResult
            training summary
        """

        self.model = model.to(self.device)
        if self.experiment_tracker is None:
            tracker_config = ExperimentConfig(
                run_name=model.__class__.__name__.lower(),
                dataset_name=train_set.metadata_frame["dataset_name"].iloc[0] if "dataset_name" in train_set.metadata_frame.columns else "unknown",
                model_name=model.__class__.__name__,
                optimizer_name="Adam",
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                sampling_strategy="chronological",
                prediction_mode="direct",
                model_hyperparameters=model.get_monitor_state(),
            )
            self.experiment_tracker = ExperimentTracker(Path("outputs") / "experiments", tracker_config)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = self._build_criterion(train_set.task_type)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if valid_set is not None else None

        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(1, self.max_epochs + 1):
            train_logs = self._run_epoch(epoch, train_loader, optimizer, criterion, train_mode=True)
            if valid_loader is not None:
                val_logs = self._run_epoch(epoch, valid_loader, optimizer, criterion, train_mode=False)
            else:
                val_logs = {"val_loss": train_logs["train_loss"]}

            epoch_logs = {**train_logs, **val_logs}
            self.history.append(epoch_logs)
            monitored_metric = float(epoch_logs["val_loss"])
            if monitored_metric < self.best_metric:
                self.best_metric = monitored_metric
                self.best_epoch = epoch
                self.best_state_dict = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

            for callback in self.callbacks:
                callback.on_validation_end(self, epoch, epoch_logs)
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_logs)

            if self.should_stop:
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        final_logs = {"best_epoch": self.best_epoch, "best_metric": self.best_metric}
        for callback in self.callbacks:
            callback.on_train_end(self, final_logs)

        return TrainingResult(best_epoch=self.best_epoch, best_metric=self.best_metric, history=self.history)

    def _run_epoch(
        self,
        epoch: int,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        *,
        train_mode: bool,
    ) -> dict[str, float]:
        phase_name = "train" if train_mode else "val"
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        loss_values: list[float] = []
        prediction_values: list[float] = []
        target_values: list[float] = []

        for batch_index, batch in enumerate(data_loader):
            inputs = self._select_inputs(batch)
            targets = batch["targets"].to(self.device)
            inputs = inputs.to(self.device)

            with torch.set_grad_enabled(train_mode):
                output_values = self.model(inputs)["prediction"]
                predictions, loss_value = self._compute_loss(output_values, targets, criterion, data_loader.dataset.task_type)
                if train_mode:
                    optimizer.zero_grad()
                    loss_value.backward()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm))
                    optimizer.step()
                    self.global_step += 1
                    batch_logs = {
                        "loss": float(loss_value.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "global_step": self.global_step,
                    }
                    for callback in self.callbacks:
                        callback.on_batch_end(self, batch_index, batch_logs)
                else:
                    gradient_norm = 0.0

            loss_values.append(float(loss_value.detach().cpu()))
            prediction_values.extend(predictions.detach().cpu().reshape(-1).tolist())
            target_values.extend(targets.detach().cpu().reshape(-1).tolist())

        epoch_loss = float(np.mean(loss_values)) if loss_values else 0.0
        metric_name = f"{phase_name}_loss"
        result = {metric_name: epoch_loss}
        if data_loader.dataset.task_type == "classification":
            predicted_classes = np.asarray(prediction_values).round().astype(int)
            true_classes = np.asarray(target_values).astype(int)
            result[f"{phase_name}_accuracy"] = float(np.mean(predicted_classes == true_classes))
        else:
            prediction_array = np.asarray(prediction_values, dtype=float)
            target_array = np.asarray(target_values, dtype=float)
            result[f"{phase_name}_rmse"] = float(np.sqrt(np.mean(np.square(prediction_array - target_array))))
        return result

    def _select_inputs(self, batch: dict[str, Any]) -> torch.Tensor:
        if getattr(self.model, "input_kind", "sequence") == "feature" and "features" in batch:
            return batch["features"]
        return batch["inputs"]

    def _build_criterion(self, task_type: str) -> nn.Module:
        if task_type == "classification":
            return nn.CrossEntropyLoss()
        return nn.SmoothL1Loss()

    def _compute_loss(
        self,
        output_values: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        task_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task_type == "classification":
            logits = output_values
            targets = targets.long()
            loss_value = criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1).float()
            return predictions, loss_value
        predictions = output_values.squeeze(-1)
        targets = targets.float().view_as(predictions)
        loss_value = criterion(predictions, targets)
        return predictions, loss_value

    def record_alert(self, alert_record: dict[str, Any]) -> None:
        """
        persist a training alert

        Parameters
        ----------
        alert_record : dict[str, Any]
            alert record
        """

        if self.experiment_tracker is not None:
            self.experiment_tracker.log_alert(alert_record)


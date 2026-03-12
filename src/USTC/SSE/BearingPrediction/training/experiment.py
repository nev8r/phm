"""
Experiment tracking module

this file is for recording experiment configuration, history and artifacts

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from USTC.SSE.BearingPrediction.common.serialization import ArtifactSerializer


@dataclass
class ExperimentConfig:
    """
    Experiment metadata and hyperparameter definition.

    Parameters
    ----------
    run_name : str
        run name
    dataset_name : str
        dataset name
    model_name : str
        model name
    optimizer_name : str
        optimizer name
    learning_rate : float
        learning rate
    weight_decay : float
        regularization coefficient
    max_epochs : int
        maximum epoch count
    batch_size : int
        batch size
    sampling_strategy : str
        sampling strategy
    prediction_mode : str
        prediction mode
    model_hyperparameters : dict[str, Any]
        model specific settings
    preprocessing_config : dict[str, Any]
        preprocessing settings
    callback_config : dict[str, Any]
        callback settings
    """

    run_name: str
    dataset_name: str
    model_name: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    sampling_strategy: str
    prediction_mode: str
    model_hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    callback_config: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """
    Persist experiment configuration, epoch history and outputs.
    """

    def __init__(self, base_dir: str | Path, config: ExperimentConfig) -> None:
        self.base_dir = Path(base_dir)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.base_dir / f"{timestamp}-{config.run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.history_records: list[dict[str, Any]] = []
        self.alert_records: list[dict[str, Any]] = []
        self.save_config()

    def save_config(self) -> None:
        """
        save experiment configuration to yaml and json
        """

        config_object = asdict(self.config)
        ArtifactSerializer.save_object(config_object, self.run_dir / "config.json")
        if ArtifactSerializer.supports_yaml():
            ArtifactSerializer.save_object(config_object, self.run_dir / "config.yaml")

    def log_epoch(self, epoch_record: dict[str, Any]) -> None:
        """
        append one epoch record and flush csv

        Parameters
        ----------
        epoch_record : dict[str, Any]
            epoch metrics
        """

        self.history_records.append(epoch_record)
        history_frame = pd.DataFrame.from_records(self.history_records)
        ArtifactSerializer.save_dataframe(history_frame, self.run_dir / "history.csv")
        ArtifactSerializer.save_dataframe(history_frame, self.run_dir / "history.pkl")

    def log_alert(self, alert_record: dict[str, Any]) -> None:
        """
        persist gradient or training alerts

        Parameters
        ----------
        alert_record : dict[str, Any]
            alert information
        """

        self.alert_records.append(alert_record)
        ArtifactSerializer.save_object(self.alert_records, self.run_dir / "alerts.json")

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """
        save final evaluation metrics

        Parameters
        ----------
        metrics : dict[str, Any]
            metric dictionary
        """

        ArtifactSerializer.save_object(metrics, self.run_dir / "metrics.json")
        if ArtifactSerializer.supports_yaml():
            ArtifactSerializer.save_object(metrics, self.run_dir / "metrics.yaml")

    def save_predictions(self, prediction_frame: pd.DataFrame) -> None:
        """
        save predictions to csv and pickle

        Parameters
        ----------
        prediction_frame : pd.DataFrame
            prediction dataframe
        """

        ArtifactSerializer.save_dataframe(prediction_frame, self.run_dir / "predictions.csv")
        ArtifactSerializer.save_dataframe(prediction_frame, self.run_dir / "predictions.pkl")


"""
Training callback module

this file is for implementing extensible epoch level callbacks

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


class Callback:
    """
    Base callback interface.
    """

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_batch_end(self, trainer: Any, batch_index: int, logs: dict[str, Any]) -> None:
        pass

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_train_end(self, trainer: Any, logs: dict[str, Any]) -> None:
        pass


@dataclass
class EarlyStopping(Callback):
    """
    Stop training early when monitored metric stops improving.
    """

    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"

    def __post_init__(self) -> None:
        self.best_value: float | None = None
        self.bad_epoch_count = 0

    def on_validation_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        current_value = float(logs[self.monitor])
        if self.best_value is None:
            self.best_value = current_value
            return

        is_improved = current_value < (self.best_value - self.min_delta) if self.mode == "min" else current_value > (self.best_value + self.min_delta)
        if is_improved:
            self.best_value = current_value
            self.bad_epoch_count = 0
            return

        self.bad_epoch_count += 1
        if self.bad_epoch_count >= self.patience:
            trainer.should_stop = True


class TensorBoardCallback(Callback):
    """
    Write training curves, gradients and text summaries to TensorBoard.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.writer: Any | None = None

    @staticmethod
    def is_available() -> bool:
        """
        report whether tensorboard support is installed

        Returns
        -------
        bool
            True when SummaryWriter can be imported
        """

        return SummaryWriter is not None

    def on_train_start(self, trainer: Any) -> None:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoardCallback requires the optional dependency tensorboard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.writer.add_text("model", trainer.model.get_monitor_state()["model_structure"])

    def on_batch_end(self, trainer: Any, batch_index: int, logs: dict[str, Any]) -> None:
        if self.writer is None:
            return
        global_step = logs["global_step"]
        self.writer.add_scalar("train/batch_loss", float(logs["loss"]), global_step)
        if "gradient_norm" in logs:
            self.writer.add_scalar("train/gradient_norm", float(logs["gradient_norm"]), global_step)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        if self.writer is None:
            return
        for metric_name, metric_value in logs.items():
            self.writer.add_scalar(f"epoch/{metric_name}", float(metric_value), epoch)
        for name, parameter in trainer.model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", parameter.detach().cpu().numpy(), epoch)

    def on_train_end(self, trainer: Any, logs: dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.close()


@dataclass
class GradientAlertCallback(Callback):
    """
    Detect gradient explosion or vanishing and emit alerts.
    """

    vanish_threshold: float = 1e-7
    explode_threshold: float = 100.0
    warmup_steps: int = 5

    def on_batch_end(self, trainer: Any, batch_index: int, logs: dict[str, Any]) -> None:
        global_step = int(logs["global_step"])
        gradient_norm = float(logs.get("gradient_norm", 0.0))
        if global_step < self.warmup_steps:
            return
        if gradient_norm < self.vanish_threshold:
            trainer.record_alert(
                {
                    "type": "gradient_vanishing",
                    "global_step": global_step,
                    "gradient_norm": gradient_norm,
                }
            )
        elif gradient_norm > self.explode_threshold:
            trainer.record_alert(
                {
                    "type": "gradient_explosion",
                    "global_step": global_step,
                    "gradient_norm": gradient_norm,
                }
            )


class ExperimentLoggerCallback(Callback):
    """
    Mirror training state into the experiment tracker.
    """

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, Any]) -> None:
        trainer.experiment_tracker.log_epoch({"epoch": epoch, **logs})

    def on_train_start(self, trainer: Any) -> None:
        callback_summary = [callback.__class__.__name__ for callback in trainer.callbacks]
        tracker_config = trainer.experiment_tracker.config
        tracker_config.callback_config = {
            **tracker_config.callback_config,
            "callbacks": callback_summary,
        }
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                tracker_config.callback_config.update(
                    {
                        "early_stopping_monitor": callback.monitor,
                        "early_stopping_patience": callback.patience,
                        "early_stopping_min_delta": callback.min_delta,
                        "early_stopping_mode": callback.mode,
                    }
                )
            elif isinstance(callback, GradientAlertCallback):
                tracker_config.callback_config.update(
                    {
                        "gradient_vanish_threshold": callback.vanish_threshold,
                        "gradient_explode_threshold": callback.explode_threshold,
                        "gradient_warmup_steps": callback.warmup_steps,
                    }
                )
            elif isinstance(callback, TensorBoardCallback):
                tracker_config.callback_config["tensorboard_log_dir"] = str(callback.log_dir)
        trainer.experiment_tracker.save_config()

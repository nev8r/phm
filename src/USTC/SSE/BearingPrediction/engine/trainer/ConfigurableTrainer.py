"""
Task-based configurable trainer.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.checkpoint.CheckpointManager import CheckpointManager
from USTC.SSE.BearingPrediction.infra.loss.LossRegistry import LossRegistry
from USTC.SSE.BearingPrediction.infra.metric.MetricRegistry import MetricRegistry
from USTC.SSE.BearingPrediction.infra.optim.OptimizerRegistry import OptimizerRegistry
from USTC.SSE.BearingPrediction.infra.optim.SchedulerRegistry import SchedulerRegistry
from USTC.SSE.BearingPrediction.infra.predict.PredictionFrame import PREDICTION_METADATA_COLUMNS, build_prediction_frame
from USTC.SSE.BearingPrediction.infra.predict.PredictionStore import PredictionStore
from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, REGRESSION
from USTC.SSE.BearingPrediction.infra.train.History import History
from USTC.SSE.BearingPrediction.infra.train.SeedManager import set_seed
from USTC.SSE.BearingPrediction.infra.train.TrainerState import TrainerState
from USTC.SSE.BearingPrediction.util.Device import select_torch_device


class ConfigurableTrainer:
    def __init__(self, cfg: DictConfig, context, datamodule, model: torch.nn.Module, model_spec: Dict):
        self.cfg = cfg
        self.trainer_cfg = OmegaConf.select(cfg, "trainer", default=cfg)
        self.context = context
        self.datamodule = datamodule
        self.model = model
        self.model_spec = model_spec
        self.task_type = str(datamodule.task_spec["task_type"])
        self.state = TrainerState()
        self.history = History()
        self.device = _select_device(str(OmegaConf.select(self.trainer_cfg, "device", default="auto")))
        self.dtype = torch.float32
        self.checkpoints = CheckpointManager(
            context.artifacts,
            enabled=bool(OmegaConf.select(self.trainer_cfg, "checkpoint.enabled", default=True)),
        )

    def train(self) -> List[Dict]:
        if self.datamodule.train is None:
            raise ValueError("Training requires a non-empty train split")
        seed = int(OmegaConf.select(self.trainer_cfg, "seed", default=OmegaConf.select(self.cfg, "project.seed", default=42)))
        set_seed(seed)
        self.model.to(self.device)
        loss_fn = LossRegistry.build(OmegaConf.select(self.trainer_cfg, "loss", default={}), self.task_type)
        optimizer = OptimizerRegistry.build(OmegaConf.select(self.trainer_cfg, "optimizer", default={}), self.model.parameters())
        scheduler = SchedulerRegistry.build(OmegaConf.select(self.trainer_cfg, "scheduler", default={}), optimizer)
        start_epoch = self._load_resume_if_requested(optimizer, scheduler)
        max_epochs = int(OmegaConf.select(self.trainer_cfg, "max_epochs", default=1))

        for epoch in range(start_epoch, max_epochs + 1):
            self.state.epoch = epoch
            train_loss = self._train_one_epoch(loss_fn, optimizer)
            row = {"epoch": epoch, "train_loss": train_loss, "lr": _learning_rate(optimizer)}

            if self.datamodule.val is not None and epoch % int(OmegaConf.select(self.trainer_cfg, "eval.every_n_epochs", default=1)) == 0:
                val_predictions, val_metrics = self.evaluate_split("val", loss_fn=loss_fn, save=False)
                del val_predictions
                row.update({f"val_{key}": value for key, value in val_metrics.items()})

            if bool(OmegaConf.select(self.trainer_cfg, "console_log.enabled", default=True)):
                print(_epoch_log(row, max_epochs), flush=True)

            if scheduler is not None:
                scheduler.step()
            self.history.append(row)
            self._save_last(optimizer, scheduler)
            if self._is_best(row):
                self.state.best_metric = float(self._monitor_value(row))
                self.state.best_epoch = epoch
                self._save_best(optimizer, scheduler)

        self._write_training_artifacts()
        self._final_evaluate_and_save(loss_fn)
        return self.history.to_list()

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict[str, Dict]:
        loss_fn = LossRegistry.build(OmegaConf.select(self.trainer_cfg, "loss", default={}), self.task_type)
        checkpoint = self.checkpoints.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return self._final_evaluate_and_save(loss_fn, eval_only=True)

    def _train_one_epoch(self, loss_fn, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        loader = self.datamodule.to_dataloader(
            "train",
            batch_size=int(OmegaConf.select(self.trainer_cfg, "batch_size", default=16)),
            shuffle=True,
            num_workers=int(OmegaConf.select(self.trainer_cfg, "num_workers", default=0)),
        )
        for batch in loader:
            x, y = self._batch_xy(batch)
            optimizer.zero_grad()
            output = self.model(x)
            loss = loss_fn(output, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf")
            loss.backward()
            clip_norm = OmegaConf.select(self.trainer_cfg, "gradient.clip_norm", default=None)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(clip_norm))
            optimizer.step()
            self.state.global_step += 1
            total_loss += float(loss.detach().cpu())
            batches += 1
        return total_loss / max(batches, 1)

    def evaluate_split(self, split: str, loss_fn=None, save: bool = False):
        dataset = self.datamodule.splits().get(split)
        if dataset is None:
            return None, {}
        loss_fn = loss_fn or LossRegistry.build(OmegaConf.select(self.trainer_cfg, "loss", default={}), self.task_type)
        self.model.eval()
        raw_outputs = []
        y_values = []
        metadata: List[Dict] = []
        losses: List[float] = []
        loader = self.datamodule.to_dataloader(
            split,
            batch_size=int(OmegaConf.select(self.trainer_cfg, "batch_size", default=16)),
            shuffle=False,
            num_workers=int(OmegaConf.select(self.trainer_cfg, "num_workers", default=0)),
        )
        with torch.no_grad():
            for batch in loader:
                x, y = self._batch_xy(batch)
                output = self.model(x)
                losses.append(float(loss_fn(output, y).detach().cpu()))
                raw_outputs.append(output.detach().cpu().numpy())
                y_values.append(y.detach().cpu().numpy())
                metadata.extend(_metadata(batch))

        y_true = np.concatenate(y_values, axis=0)
        raw_output = np.concatenate(raw_outputs, axis=0)
        predictions = build_prediction_frame(
            metadata=metadata,
            y_true=y_true,
            raw_output=raw_output,
            task_type=self.task_type,
            target_columns=self.datamodule.target_columns,
        )
        metrics = self._metrics_from_predictions(predictions, y_true, raw_output)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        if save:
            self._save_split_outputs(split, predictions, metrics)
        return predictions, metrics

    def _metrics_from_predictions(self, predictions, y_true, raw_output) -> Dict[str, float]:
        metric_fn = MetricRegistry.build(self.task_type)
        if self.task_type == REGRESSION:
            return metric_fn(y_true, raw_output)
        if self.task_type in CLASSIFICATION_TYPES:
            return metric_fn(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy())
        return {}

    def _batch_xy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["x"].to(self.device, dtype=self.dtype)
        if self.task_type in CLASSIFICATION_TYPES:
            y = batch["y"].to(self.device, dtype=torch.long).reshape(-1)
        else:
            y = batch["y"].to(self.device, dtype=self.dtype)
        return x, y

    def _save_split_outputs(self, split: str, predictions, metrics: Dict[str, float]) -> None:
        PredictionStore(
            self.context.artifacts,
            write_csv=bool(OmegaConf.select(self.trainer_cfg, "prediction.write_csv", default=False)),
        ).save(split, predictions)
        self.context.artifacts.write_json(f"metrics/{split}_metrics.json", metrics)

    def _final_evaluate_and_save(self, loss_fn, eval_only: bool = False) -> Dict[str, Dict]:
        outputs: Dict[str, Dict] = {}
        save_cfg = OmegaConf.select(self.trainer_cfg, "prediction", default={})
        split_flags = {
            "train": bool(OmegaConf.select(save_cfg, "save_train", default=False)),
            "val": bool(OmegaConf.select(save_cfg, "save_val", default=True)),
            "test": bool(OmegaConf.select(save_cfg, "save_test", default=True)),
        }
        if eval_only:
            split_flags = {name: dataset is not None for name, dataset in self.datamodule.splits().items()}
        for split, should_save in split_flags.items():
            if not should_save:
                continue
            if self.datamodule.splits().get(split) is not None:
                _, metrics = self.evaluate_split(split, loss_fn=loss_fn, save=True)
                outputs[split] = metrics
            else:
                empty = pd.DataFrame(columns=PREDICTION_METADATA_COLUMNS)
                self._save_split_outputs(split, empty, {})
                outputs[split] = {}
        return outputs

    def _write_training_artifacts(self) -> None:
        self.context.artifacts.write_json("metrics/history.json", self.history.to_list())
        self.context.artifacts.write_json("trainer/trainer_state.json", self.state.to_dict())
        self.context.artifacts.write_text("trainer/model_summary.txt", str(self.model) + "\n")
        self.context.artifacts.write_text("report.md", _report_markdown(self.model_spec, self.datamodule.task_spec, self.state))

    def _checkpoint_payload(self, optimizer, scheduler) -> Dict:
        return {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model": self.model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "model_spec": self.model_spec,
            "task_spec": self.datamodule.task_spec,
            "trainer_config": OmegaConf.to_container(self.trainer_cfg, resolve=True),
            "feature_columns": self.datamodule.feature_columns,
            "target_columns": self.datamodule.target_columns,
            "history": self.history.to_list(),
        }

    def _save_last(self, optimizer, scheduler) -> None:
        if bool(OmegaConf.select(self.trainer_cfg, "checkpoint.save_last", default=True)):
            self.checkpoints.save_last(self._checkpoint_payload(optimizer, scheduler))

    def _save_best(self, optimizer, scheduler) -> None:
        if bool(OmegaConf.select(self.trainer_cfg, "checkpoint.save_best", default=True)):
            self.checkpoints.save_best(self._checkpoint_payload(optimizer, scheduler))

    def _is_best(self, row: Dict) -> bool:
        value = self._monitor_value(row)
        if value is None:
            return self.state.best_metric is None
        if self.state.best_metric is None:
            return True
        mode = str(OmegaConf.select(self.trainer_cfg, "monitor.mode", default="min"))
        return value < self.state.best_metric if mode == "min" else value > self.state.best_metric

    def _monitor_value(self, row: Dict) -> Optional[float]:
        split = str(OmegaConf.select(self.trainer_cfg, "monitor.split", default="val"))
        metric = str(OmegaConf.select(self.trainer_cfg, "monitor.metric", default="loss"))
        key = f"{split}_{metric}"
        if key in row:
            return float(row[key])
        if metric == "loss" and "train_loss" in row:
            return float(row["train_loss"])
        return None

    def _load_resume_if_requested(self, optimizer, scheduler) -> int:
        resume_from = OmegaConf.select(self.trainer_cfg, "checkpoint.resume_from", default=None)
        if resume_from in (None, "null", ""):
            return 1
        checkpoint = self.checkpoints.load(resume_from)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.best_metric = checkpoint.get("best_metric")
        self.state.best_epoch = checkpoint.get("best_epoch")
        for row in checkpoint.get("history", []):
            self.history.append(row)
        return int(checkpoint["epoch"]) + 1


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return select_torch_device()
    return torch.device(name)


def _learning_rate(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _epoch_log(row: Dict, max_epochs: int) -> str:
    parts = [
        f"[train] epoch {int(row['epoch'])}/{max_epochs}",
        f"train_loss={float(row['train_loss']):.6f}",
        f"lr={float(row['lr']):.2e}",
    ]
    for key in sorted(row):
        if key.startswith("val_"):
            value = row[key]
            if isinstance(value, (int, float, np.floating)):
                parts.append(f"{key}={float(value):.6f}")
    return " | ".join(parts)


def _metadata(batch) -> List[Dict]:
    size = len(batch["example_uid"])
    rows: List[Dict] = []
    for index in range(size):
        rows.append({
            "example_uid": _item(batch["example_uid"], index),
            "split": _item(batch["split"], index),
            "sample_uid": _item(batch["sample_uid"], index),
            "target_sample_uid": _item(batch["target_sample_uid"], index),
            "dataset": _item(batch["dataset"], index),
            "bearing_id": _item(batch["bearing_id"], index),
            "condition_id": _item(batch["condition_id"], index),
            "target_timestep": int(_item(batch["target_timestep"], index)),
        })
    return rows


def _item(value, index):
    if isinstance(value, torch.Tensor):
        return value[index].item()
    return value[index]


def _report_markdown(model_spec: Dict, task_spec: Dict, state: TrainerState) -> str:
    return "\n".join([
        "# Training Report",
        "",
        f"- model: {model_spec.get('name')}",
        f"- task: {task_spec.get('name')}",
        f"- best_metric: {state.best_metric}",
        f"- best_epoch: {state.best_epoch}",
        "",
    ])

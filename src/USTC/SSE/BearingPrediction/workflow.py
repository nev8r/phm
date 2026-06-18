"""
Workflow utilities for paper reproduction jobs

this file is for preparing data, training paper models, and benchmarking baselines

created by zdh

copyright USTC

2026
"""

from __future__ import annotations

import csv
import copy
import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader

from USTC.SSE.BearingPrediction.analysis import build_domain_feature_names, write_json
from USTC.SSE.BearingPrediction.data import Dataset
from USTC.SSE.BearingPrediction.data.paper import (
    PHM2012_FULL_TEST_BEARINGS,
    PHM2012_LEARNING_BEARINGS,
    SequenceFeatureDataset,
    XJTU_HEALTH_STATES,
    build_phm2012_rul_feature_cache,
    build_xjtu_binary_fault_diagnosis_feature_cache,
    fit_feature_standardizer,
    load_feature_cache,
    make_sequence_index,
)
from USTC.SSE.BearingPrediction.engine.trainer.BaseTrainer import BaseTrainer
from USTC.SSE.BearingPrediction.engine.callback.ABCTrainCallback import ABCTrainCallback
from USTC.SSE.BearingPrediction.model.paper import (
    PaperCBAMCNNLSTMRegressor,
    ResCNNLSTMClassifier,
)
from USTC.SSE.BearingPrediction.util.Device import select_torch_device


PAPER_RUL_TEST_BEARINGS = (
    "Bearing1_3",
    "Bearing1_4",
    "Bearing1_5",
    "Bearing1_7",
    "Bearing2_3",
    "Bearing2_6",
)


@dataclass
class FeatureCache:
    task: str
    features: np.ndarray
    targets: np.ndarray
    ranges: dict[str, tuple[int, int]]
    feature_names: list[str]
    metadata: dict[str, Any]


@dataclass
class PreparedData:
    task: str
    feature_cache: FeatureCache
    sequence_length: int
    batch_size: int
    mean: np.ndarray
    std: np.ndarray
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray
    train_dataset: SequenceFeatureDataset
    val_dataset: SequenceFeatureDataset
    test_dataset: SequenceFeatureDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def find_project_root(start: Path | None = None) -> Path:
    cursor = Path.cwd() if start is None else Path(start)
    for path in (cursor, *cursor.parents):
        if (path / "data" / "loader_roots").exists() or (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("could not locate project root")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return select_torch_device()
    return torch.device(requested)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_metadata(raw: dict[str, Any]) -> tuple[int, bool, int]:
    metadata = raw.get("metadata", {})
    fft_bins = int(metadata.get("fft_bins", 256))
    include_handcrafted = bool(metadata.get("include_handcrafted", True))
    per_channel = fft_bins + (20 if include_handcrafted else 0)
    feature_count = int(raw["features"].shape[1])
    channel_count = max(1, feature_count // max(1, per_channel))
    return fft_bins, include_handcrafted, channel_count


def _load_full_feature_cache(task: str, project_root: Path) -> FeatureCache:
    cache_dir = project_root / "cache" / "paper_features"
    if task == "rul":
        cache_path = cache_dir / "phm2012_rul_fft256_full.npz"
        if not cache_path.exists():
            build_phm2012_rul_feature_cache(
                project_root / "data" / "loader_roots" / "phm2012",
                cache_path,
                fft_bins=256,
                include_handcrafted=True,
            )
    elif task == "fault":
        cache_path = cache_dir / "xjtu_binary_fault_diagnosis_fft256_full.npz"
        if not cache_path.exists():
            build_xjtu_binary_fault_diagnosis_feature_cache(
                project_root / "data" / "loader_roots" / "xjtu",
                cache_path,
                fft_bins=256,
                include_handcrafted=True,
            )
    else:
        raise ValueError(f"unsupported task: {task}")

    raw = load_feature_cache(cache_path)
    fft_bins, include_handcrafted, channel_count = _parse_metadata(raw)
    feature_names = build_domain_feature_names(
        fft_bins=fft_bins,
        include_handcrafted=include_handcrafted,
        channel_count=channel_count,
    )
    if len(feature_names) != raw["features"].shape[1]:
        feature_names = [f"feature_{index}" for index in range(raw["features"].shape[1])]
    return FeatureCache(
        task=task,
        features=raw["features"].astype(np.float32),
        targets=raw["targets"].astype(np.float32),
        ranges=raw["ranges"],
        feature_names=feature_names,
        metadata={**raw.get("metadata", {}), "source": str(cache_path), "mode": "full-cache"},
    )


def _make_sample_feature_cache(task: str, seed: int = 42) -> FeatureCache:
    rng = np.random.default_rng(seed if task == "rul" else seed + 1)
    if task == "rul":
        bearing_count = 4
        rows_per_bearing = 52
        feature_count = 96
        ranges: dict[str, tuple[int, int]] = {}
        features = []
        targets = []
        for bearing_index in range(bearing_count):
            start = len(features)
            progress = np.linspace(0.0, 1.0, rows_per_bearing)
            basis = np.stack(
                [
                    progress,
                    progress**2,
                    np.sin(progress * np.pi),
                    np.cos(progress * np.pi),
                ],
                axis=1,
            )
            weights = rng.normal(0.0, 0.25, size=(basis.shape[1], feature_count))
            bearing_features = basis @ weights + rng.normal(0.0, 0.03, size=(rows_per_bearing, feature_count))
            bearing_features[:, 0] = 0.4 + 0.8 * progress
            bearing_features[:, 1] = 0.2 + 1.2 * progress**2
            features.extend(bearing_features.astype(np.float32))
            targets.extend((1.0 - progress).reshape(-1, 1).astype(np.float32))
            ranges[f"SampleRUL_{bearing_index + 1}"] = (start, len(features))
        names = [f"sample_feature_{index:03d}" for index in range(feature_count)]
        names[0] = "rms"
        names[1] = "band_energy_3_6khz"
    elif task == "fault":
        bearing_count = 5
        rows_per_bearing = 48
        feature_count = 64
        ranges = {}
        features = []
        targets = []
        for bearing_index in range(bearing_count):
            start = len(features)
            progress = np.linspace(0.0, 1.0, rows_per_bearing)
            labels = (progress >= 0.58 + 0.03 * (bearing_index % 2)).astype(np.int64)
            base = rng.normal(0.0, 0.10, size=(rows_per_bearing, feature_count))
            base[:, 0] = 0.25 + 1.1 * progress + rng.normal(0.0, 0.03, rows_per_bearing)
            base[:, 1] = labels * 1.2 + rng.normal(0.0, 0.08, rows_per_bearing)
            base[:, 2] = 0.3 + labels * 0.9 + rng.normal(0.0, 0.05, rows_per_bearing)
            features.extend(base.astype(np.float32))
            targets.extend(labels.reshape(-1, 1).astype(np.float32))
            ranges[f"SampleFault_{bearing_index + 1}"] = (start, len(features))
        names = [f"sample_feature_{index:03d}" for index in range(feature_count)]
        names[0] = "rms"
        names[1] = "fault_energy"
        names[2] = "kurtosis"
    else:
        raise ValueError(f"unsupported task: {task}")

    return FeatureCache(
        task=task,
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        ranges=ranges,
        feature_names=names,
        metadata={"source": "deterministic-sample", "mode": "sample"},
    )


def load_feature_cache_for_task(task: str, *, sample: bool = False, project_root: Path | None = None) -> FeatureCache:
    if sample:
        return _make_sample_feature_cache(task)
    return _load_full_feature_cache(task, project_root or find_project_root())


def prepare_sequence_data(
    task: str,
    *,
    sample: bool = False,
    sequence_length: int,
    batch_size: int,
    seed: int = 42,
) -> PreparedData:
    cache = load_feature_cache_for_task(task, sample=sample)
    if task == "rul":
        if sample:
            bearings = list(cache.ranges)
            train_bearings = bearings[:-1]
            test_bearings = bearings[-1:]
        else:
            train_bearings = list(PHM2012_LEARNING_BEARINGS)
            test_bearings = [bearing for bearing in PAPER_RUL_TEST_BEARINGS if bearing in cache.ranges]
        train_val_windows, _ = make_sequence_index(
            cache.ranges,
            sequence_length=sequence_length,
            sequence_step=1,
            bearings=train_bearings,
        )
        test_windows, _ = make_sequence_index(
            cache.ranges,
            sequence_length=sequence_length,
            sequence_step=1,
            bearings=test_bearings,
        )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_val_windows))
        val_size = max(int(len(order) * 0.15), 1)
        val_windows = train_val_windows[order[:val_size]]
        train_windows = train_val_windows[order[val_size:]]
    elif task == "fault":
        windows, _ = make_sequence_index(
            cache.ranges,
            sequence_length=sequence_length,
            sequence_step=1,
        )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(windows))
        test_size = max(int(len(order) * 0.15), 1)
        val_size = max(int(len(order) * 0.15), 1)
        test_windows = windows[order[:test_size]]
        val_windows = windows[order[test_size:test_size + val_size]]
        train_windows = windows[order[test_size + val_size:]]
    else:
        raise ValueError(f"unsupported task: {task}")

    mean, std = fit_feature_standardizer(cache.features, train_windows)
    train_dataset = SequenceFeatureDataset(cache.features, cache.targets, train_windows, mean=mean, std=std)
    val_dataset = SequenceFeatureDataset(cache.features, cache.targets, val_windows, mean=mean, std=std)
    test_dataset = SequenceFeatureDataset(cache.features, cache.targets, test_windows, mean=mean, std=std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return PreparedData(
        task=task,
        feature_cache=cache,
        sequence_length=sequence_length,
        batch_size=batch_size,
        mean=mean,
        std=std,
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    nonzero = y_true != 0
    if np.any(nonzero):
        percent_error = (y_true[nonzero] - y_pred[nonzero]) * 100.0 / y_true[nonzero]
        phm_scores = np.empty_like(percent_error, dtype=np.float64)
        early_mask = percent_error <= 0
        phm_scores[early_mask] = np.exp(-np.log(0.5) * (percent_error[early_mask] / 5.0))
        phm_scores[~early_mask] = np.exp(np.log(0.5) * (percent_error[~early_mask] / 20.0))
        phm_score = float(np.mean(phm_scores))
    else:
        phm_score = 0.0
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "phm2012_score": phm_score,
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fault_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _preset_config(task: str, preset: str, sample: bool) -> dict[str, Any]:
    if task == "rul":
        config = {
            "epochs": 200,
            "sequence_length": 32,
            "batch_size": 128,
            "learning_rate": 7e-4,
            "weight_decay": 1e-4,
        }
        if preset == "smoke" or sample:
            config.update({"epochs": 1, "sequence_length": 8, "batch_size": 16, "learning_rate": 1e-3})
    elif task == "fault":
        config = {
            "epochs": 35,
            "sequence_length": 8,
            "batch_size": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        }
        if preset == "smoke" or sample:
            config.update({"epochs": 1, "sequence_length": 8, "batch_size": 16, "learning_rate": 1e-3})
    else:
        raise ValueError(f"unsupported task: {task}")
    return config


def _build_model(task: str, input_dim: int, *, smoke: bool) -> torch.nn.Module:
    if task == "rul":
        return PaperCBAMCNNLSTMRegressor(
            input_dim=input_dim,
            lstm_hidden=32 if smoke else 160,
            lstm_layers=1 if smoke else 2,
            cbam_reduction=8 if smoke else 16,
            cbam_kernel_size=7,
            dropout=0.10 if smoke else 0.15,
        )
    if task == "fault":
        return ResCNNLSTMClassifier(
            input_dim=input_dim,
            num_classes=len(XJTU_HEALTH_STATES),
            hidden_dim=24 if smoke else 64,
            conv_channels=24 if smoke else 64,
            lstm_layers=1,
            residual_blocks=1 if smoke else 2,
            dropout=0.10 if smoke else 0.20,
        )
    raise ValueError(f"unsupported task: {task}")


def _evaluate_rul_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            predictions.append(y_hat.cpu().numpy())
            labels.append(y.cpu().numpy())
            losses.append(loss.item() * x.size(0))
    pred = np.vstack(predictions)
    true = np.vstack(labels)
    metrics = regression_metrics(true, pred)
    metrics["loss"] = float(sum(losses) / len(true))
    return metrics, pred.reshape(-1), true.reshape(-1)


def _evaluate_fault_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    logits_list = []
    labels_list = []
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long).reshape(-1)
            logits = model(x)
            loss = criterion(logits, y)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            losses.append(loss.item() * x.size(0))
    logits = np.vstack(logits_list)
    labels = np.concatenate(labels_list)
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = prob.argmax(axis=1)
    metrics = classification_metrics(labels, pred)
    metrics["loss"] = float(sum(losses) / len(labels))
    return metrics, prob, pred, labels


class BestValidationModelCallback(ABCTrainCallback):
    """Keep the best validation checkpoint while still using BaseTrainer."""

    def __init__(
        self,
        *,
        task: str,
        val_loader: DataLoader,
        device: torch.device,
        criterion: torch.nn.Module,
    ):
        self.task = task
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.history: list[dict[str, float]] = []

    def on_train_begin(self, model) -> bool:
        self._evaluate_and_maybe_store(model, epoch=0)
        return True

    def on_epoch_end(self, model, epoch: int, avg_loss) -> bool:
        self._evaluate_and_maybe_store(model, epoch=epoch)
        return True

    def on_train_end(self, model) -> bool:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return True

    def _evaluate_and_maybe_store(self, model, *, epoch: int) -> None:
        if self.task == "rul":
            metrics, _, _ = _evaluate_rul_model(model, self.val_loader, self.device, self.criterion)
        else:
            metrics, _, _, _ = _evaluate_fault_model(model, self.val_loader, self.device, self.criterion)
        loss = float(metrics["loss"])
        self.history.append({"epoch": epoch, "loss": loss})
        if loss < self.best_loss:
            self.best_epoch = epoch
            self.best_loss = loss
            self.best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})


def _render_training_curve(path: Path, losses: list[float], ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if losses:
        ax.plot(np.arange(1, len(losses) + 1), losses, marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title("Training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_rul_prediction_curve(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(y_true, label="True RUL", linewidth=1.6)
    ax.plot(y_pred, label="Predicted RUL", linewidth=1.4)
    ax.set_xlabel("Test window")
    ax.set_ylabel("Normalized RUL")
    ax.set_title("PHM2012 RUL prediction curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_rul_prediction_by_bearing(path: Path, predictions: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bearing_count = len(predictions)
    rows = min(3, bearing_count)
    cols = int(np.ceil(bearing_count / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (bearing, values) in zip(axes.flat, predictions.items()):
        ax.set_axis_on()
        ax.plot(values["true"], label="True", linewidth=1.3)
        ax.plot(values["pred"], label="Pred", linewidth=1.2)
        ax.set_title(bearing)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("PHM2012 paper-bearing RUL prediction curves", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_fault_confusion_matrix(path: Path, cm: list[list[int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(
        np.asarray(cm, dtype=int),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=XJTU_HEALTH_STATES,
        yticklabels=XJTU_HEALTH_STATES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Healthy/Faulty confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_predictions(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_dict_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_benchmark_chart(path: Path, task: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["model"] for row in rows]
    if task == "rul":
        values = [float(row["rmse"]) for row in rows]
        ylabel = "RMSE"
        title = "PHM2012 RUL baseline comparison"
        color = "#1f77b4"
    else:
        values = [float(row["weighted_f1"]) for row in rows]
        ylabel = "Weighted F1"
        title = "XJTU-SY fault baseline comparison"
        color = "#2ca02c"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    if task == "fault":
        ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _rul_test_bearings(prepared: PreparedData, sample: bool) -> list[str]:
    if sample:
        return list(prepared.feature_cache.ranges)[-1:]
    return [bearing for bearing in PAPER_RUL_TEST_BEARINGS if bearing in prepared.feature_cache.ranges]


def _evaluate_rul_by_bearing(
    model: torch.nn.Module,
    prepared: PreparedData,
    device: torch.device,
    criterion: torch.nn.Module,
    bearings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions_by_bearing: dict[str, dict[str, np.ndarray]] = {}
    reference = _paper_reference("rul")["bearings"]
    for bearing in bearings:
        windows, _ = make_sequence_index(
            prepared.feature_cache.ranges,
            sequence_length=prepared.sequence_length,
            sequence_step=1,
            bearings=[bearing],
        )
        dataset = SequenceFeatureDataset(
            prepared.feature_cache.features,
            prepared.feature_cache.targets,
            windows,
            mean=prepared.mean,
            std=prepared.std,
        )
        loader = DataLoader(dataset, batch_size=prepared.batch_size, shuffle=False, num_workers=0)
        metrics, pred, true = _evaluate_rul_model(model, loader, device, criterion)
        row = {"bearing": bearing, "windows": len(dataset), **metrics}
        if bearing in reference:
            for key, value in reference[bearing].items():
                row[f"paper_{key}"] = value
                if key in metrics:
                    row[f"delta_{key}"] = metrics[key] - value
        metric_rows.append(row)
        predictions_by_bearing[bearing] = {"true": true, "pred": pred}
        prediction_rows.extend(
            [
                {"bearing": bearing, "index": index, "y_true": float(t), "y_pred": float(p)}
                for index, (t, p) in enumerate(zip(true, pred))
            ]
        )
    return metric_rows, prediction_rows, predictions_by_bearing


def run_paper_training(
    *,
    task: str,
    preset: str,
    sample: bool,
    device_name: str,
    run_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    preset_values = _preset_config(task, preset, sample)
    prepared = prepare_sequence_data(
        task,
        sample=sample,
        sequence_length=preset_values["sequence_length"],
        batch_size=preset_values["batch_size"],
        seed=seed,
    )
    device = resolve_device(device_name)
    smoke = preset == "smoke" or sample
    model = _build_model(task, prepared.feature_cache.features.shape[1], smoke=smoke)
    criterion: torch.nn.Module = nn.MSELoss() if task == "rul" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preset_values["learning_rate"]),
        weight_decay=float(preset_values["weight_decay"]),
    )
    best_callback = BestValidationModelCallback(
        task=task,
        val_loader=prepared.val_loader,
        device=device,
        criterion=criterion,
    )
    trainer = BaseTrainer(
        config={
            "device": device,
            "epochs": int(preset_values["epochs"]),
            "batch_size": int(preset_values["batch_size"]),
            "criterion": criterion,
            "optimizer": optimizer,
            "lr": float(preset_values["learning_rate"]),
            "weight_decay": float(preset_values["weight_decay"]),
            "data_loader": prepared.train_loader,
            "grad_clip_norm": 1.0,
            "callbacks": [best_callback],
        }
    )

    start = time.perf_counter()
    loss_history = trainer(model, Dataset(name=f"{task.upper()} paper model train"))
    train_seconds = time.perf_counter() - start
    train_losses = next(iter(loss_history.values()), [])

    if task == "rul":
        val_metrics, _, _ = _evaluate_rul_model(model, prepared.val_loader, device, criterion)
        start_infer = time.perf_counter()
        test_metrics, test_pred, test_true = _evaluate_rul_model(model, prepared.test_loader, device, criterion)
        inference_seconds = time.perf_counter() - start_infer
        rul_bearings = _rul_test_bearings(prepared, sample)
        bearing_metrics, prediction_rows, predictions_by_bearing = _evaluate_rul_by_bearing(
            model,
            prepared,
            device,
            criterion,
            rul_bearings,
        )
        _render_training_curve(figures_dir / "training_curve.png", train_losses, "MSE loss")
        _render_rul_prediction_curve(figures_dir / "rul_prediction_curve.png", test_true, test_pred)
        _render_rul_prediction_by_bearing(figures_dir / "rul_prediction_by_bearing.png", predictions_by_bearing)
        _write_predictions(
            run_dir / "predictions.csv",
            prediction_rows,
            ["bearing", "index", "y_true", "y_pred"],
        )
        _write_dict_rows_csv(run_dir / "rul_bearing_metrics.csv", bearing_metrics)
        model_name = "PaperCBAMCNNLSTMRegressor"
    else:
        val_metrics, _, _, _ = _evaluate_fault_model(model, prepared.val_loader, device, criterion)
        start_infer = time.perf_counter()
        test_metrics, test_prob, test_pred, test_true = _evaluate_fault_model(model, prepared.test_loader, device, criterion)
        inference_seconds = time.perf_counter() - start_infer
        _render_training_curve(figures_dir / "training_curve.png", train_losses, "Cross entropy loss")
        _render_fault_confusion_matrix(figures_dir / "fault_confusion_matrix.png", test_metrics["confusion_matrix"])
        _write_predictions(
            run_dir / "predictions.csv",
            [
                {
                    "index": index,
                    "y_true": int(true),
                    "y_pred": int(pred),
                    "prob_healthy": float(prob[0]),
                    "prob_fault": float(prob[1]),
                }
                for index, (true, pred, prob) in enumerate(zip(test_true, test_pred, test_prob))
            ],
            ["index", "y_true", "y_pred", "prob_healthy", "prob_fault"],
        )
        model_name = "ResCNNLSTMClassifier"

    checkpoint = run_dir / "model_state.pt"
    torch.save(model.state_dict(), checkpoint)
    standardizer_path = run_dir / "standardizer.npz"
    np.savez_compressed(standardizer_path, mean=prepared.mean, std=prepared.std)
    summary = {
        "model": model_name,
        "task": task,
        "parameter_count": count_parameters(model),
        "model_state_path": str(checkpoint),
        "model_state_size_bytes": checkpoint.stat().st_size,
        "standardizer_path": str(standardizer_path),
        "standardizer_size_bytes": standardizer_path.stat().st_size,
        "input_dim": int(prepared.feature_cache.features.shape[1]),
        "sequence_length": prepared.sequence_length,
    }
    config = {
        "command": "train",
        "task": task,
        "preset": preset,
        "sample": sample,
        "device": str(device),
        "seed": seed,
        "trainer": "BaseTrainer",
        "model": model_name,
        **preset_values,
        "feature_source": prepared.feature_cache.metadata,
        "split_sizes": {
            "train": len(prepared.train_dataset),
            "val": len(prepared.val_dataset),
            "test": len(prepared.test_dataset),
        },
    }
    if task == "rul":
        config["test_scope"] = "paper_mainline_bearings" if not sample else "sample_last_bearing"
        config["test_bearings"] = _rul_test_bearings(prepared, sample)
    metrics = {
        "command": "train",
        "task": task,
        "trainer": "BaseTrainer",
        "model": model_name,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "train_loss_history": train_losses,
        "validation_loss_history": best_callback.history,
        "best_validation_epoch": best_callback.best_epoch,
        "best_validation_loss": best_callback.best_loss,
        "val": val_metrics,
        "test": test_metrics,
        "paper_reference": _paper_reference(task),
    }
    if task == "rul":
        metrics["test_scope"] = config["test_scope"]
        metrics["test_bearings"] = config["test_bearings"]
        metrics["bearing_metrics"] = bearing_metrics
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "model_summary.json", summary)
    return metrics


def _paper_reference(task: str) -> dict[str, Any]:
    if task == "rul":
        return {
            "model": "CBAM-CNN-LSTM",
            "bearings": {
                "Bearing1_3": {"mse": 0.0047, "rmse": 0.069, "mae": 0.037, "r2": 0.943},
                "Bearing1_4": {"mse": 0.0077, "rmse": 0.086, "mae": 0.060, "r2": 0.910},
                "Bearing1_5": {"mse": 0.0198, "rmse": 0.141, "mae": 0.091, "r2": 0.762},
                "Bearing1_7": {"mse": 0.0137, "rmse": 0.117, "mae": 0.078, "r2": 0.836},
                "Bearing2_3": {"mse": 0.0223, "rmse": 0.148, "mae": 0.112, "r2": 0.737},
                "Bearing2_6": {"mse": 0.0085, "rmse": 0.088, "mae": 0.069, "r2": 0.906},
            },
        }
    if task == "fault":
        return {"model": "ResCNN-LSTM", "accuracy": 0.9639, "weighted_f1": 0.964}
    raise ValueError(f"unsupported task: {task}")


def _dataset_to_arrays(dataset: SequenceFeatureDataset) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for index in range(len(dataset)):
        x, y = dataset[index]
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)


def _window_summary_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    last = x[:, -1, :]
    return np.concatenate([mean, std, last], axis=1)


def _model_pickle_size_bytes(model: Any) -> int:
    try:
        return len(pickle.dumps(model))
    except Exception:
        return 0


def _rocket_panel(x: np.ndarray, max_channels: int = 96) -> tuple[np.ndarray, int]:
    if x.shape[2] > max_channels:
        selected = np.linspace(0, x.shape[2] - 1, max_channels).round().astype(int)
        x = x[:, :, selected]
    return np.transpose(x, (0, 2, 1)), int(x.shape[2])


def _parse_baselines(baselines: str, task: str) -> list[str]:
    if baselines == "all":
        return ["linear", "forest", "rocket", "deep"]
    return [item.strip() for item in baselines.split(",") if item.strip()]


def _find_latest_train_run(search_root: Path, task: str, sample: bool) -> Path | None:
    candidates = sorted(search_root.glob(f"*_train_{task}"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        config_path = candidate / "config.json"
        metrics_path = candidate / "metrics.json"
        summary_path = candidate / "model_summary.json"
        if not (config_path.exists() and metrics_path.exists() and summary_path.exists()):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if bool(config.get("sample")) == bool(sample):
            return candidate
    return None


def _deep_baseline_row_from_run(run: Path) -> dict[str, Any]:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "model_summary.json").read_text(encoding="utf-8"))
    return {
        "baseline": "deep",
        "model": metrics["model"],
        "fit_seconds": metrics["train_seconds"],
        "inference_seconds": metrics["inference_seconds"],
        "model_size_bytes": summary["model_state_size_bytes"],
        "source_run": str(run),
        **metrics["test"],
    }


def _fit_predict_baseline(
    *,
    task: str,
    baseline: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample: bool,
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    if baseline == "linear":
        if task == "rul":
            model = Ridge(alpha=1.0)
            model.fit(_window_summary_features(x_train), y_train.reshape(-1))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(_window_summary_features(x_test))
            infer_seconds = time.perf_counter() - infer_start
            metrics = regression_metrics(y_test, pred)
            model_name = "Ridge"
        else:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(_window_summary_features(x_train), y_train.reshape(-1).astype(int))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(_window_summary_features(x_test))
            infer_seconds = time.perf_counter() - infer_start
            metrics = classification_metrics(y_test, pred)
            model_name = "LogisticRegression"
    elif baseline == "forest":
        if task == "rul":
            model = RandomForestRegressor(
                n_estimators=20 if sample else 120,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(_window_summary_features(x_train), y_train.reshape(-1))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(_window_summary_features(x_test))
            infer_seconds = time.perf_counter() - infer_start
            metrics = regression_metrics(y_test, pred)
            model_name = "RandomForestRegressor"
        else:
            model = RandomForestClassifier(
                n_estimators=20 if sample else 120,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(_window_summary_features(x_train), y_train.reshape(-1).astype(int))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(_window_summary_features(x_test))
            infer_seconds = time.perf_counter() - infer_start
            metrics = classification_metrics(y_test, pred)
            model_name = "RandomForestClassifier"
    elif baseline == "rocket":
        if task == "rul":
            from sktime.regression.kernel_based import RocketRegressor

            model = RocketRegressor(num_kernels=256 if sample else 1024, random_state=seed)
            x_train_panel, channel_count = _rocket_panel(x_train)
            x_test_panel, _ = _rocket_panel(x_test, max_channels=channel_count)
            model.fit(x_train_panel, y_train.reshape(-1))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(x_test_panel)
            infer_seconds = time.perf_counter() - infer_start
            metrics = regression_metrics(y_test, pred)
            model_name = "RocketRegressor"
        else:
            from sktime.classification.kernel_based import RocketClassifier

            model = RocketClassifier(num_kernels=256 if sample else 1024, random_state=seed)
            x_train_panel, channel_count = _rocket_panel(x_train)
            x_test_panel, _ = _rocket_panel(x_test, max_channels=channel_count)
            model.fit(x_train_panel, y_train.reshape(-1).astype(int))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            pred = model.predict(x_test_panel)
            infer_seconds = time.perf_counter() - infer_start
            metrics = classification_metrics(y_test, pred)
            model_name = "RocketClassifier"
    else:
        raise ValueError(f"unsupported baseline: {baseline}")

    return {
        "baseline": baseline,
        "model": model_name,
        "fit_seconds": fit_seconds,
        "inference_seconds": infer_seconds,
        "model_size_bytes": _model_pickle_size_bytes(model),
        "rocket_channel_count": channel_count if baseline == "rocket" else "",
        **metrics,
    }


def run_benchmark(
    *,
    task: str,
    baselines: str,
    sample: bool,
    run_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["rul", "fault"] if task == "all" else [task]
    results: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for item in tasks:
        preset = _preset_config(item, "smoke" if sample else "paper", sample)
        prepared = prepare_sequence_data(
            item,
            sample=sample,
            sequence_length=preset["sequence_length"],
            batch_size=preset["batch_size"],
            seed=seed,
        )
        x_train, y_train = _dataset_to_arrays(prepared.train_dataset)
        x_test, y_test = _dataset_to_arrays(prepared.test_dataset)
        task_results = []
        for baseline in _parse_baselines(baselines, item):
            if baseline == "deep":
                latest_train_run = _find_latest_train_run(run_dir.parent, item, sample)
                if latest_train_run is not None:
                    row = _deep_baseline_row_from_run(latest_train_run)
                else:
                    deep_dir = run_dir / f"deep_{item}"
                    deep_metrics = run_paper_training(
                        task=item,
                        preset="smoke" if sample else "paper",
                        sample=sample,
                        device_name="auto",
                        run_dir=deep_dir,
                        seed=seed,
                    )
                    summary = json.loads((deep_dir / "model_summary.json").read_text(encoding="utf-8"))
                    row = {
                        "baseline": "deep",
                        "model": deep_metrics["model"],
                        "fit_seconds": deep_metrics["train_seconds"],
                        "inference_seconds": deep_metrics["inference_seconds"],
                        "model_size_bytes": summary["model_state_size_bytes"],
                        "source_run": str(deep_dir),
                        **deep_metrics["test"],
                    }
            else:
                row = _fit_predict_baseline(
                    task=item,
                    baseline=baseline,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    sample=sample,
                    seed=seed,
                )
            task_results.append(row)
            csv_rows.append({"task": item, **row})
        results[item] = task_results
        _render_benchmark_chart(figures_dir / f"{item}_benchmark.png", item, task_results)

    fieldnames = sorted({key for row in csv_rows for key in row.keys() if key != "confusion_matrix"})
    with (run_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    config = {
        "command": "benchmark",
        "task": task,
        "baselines": baselines,
        "sample": sample,
        "seed": seed,
    }
    metrics = {
        "command": "benchmark",
        "task": task,
        "baselines": baselines,
        "results": results,
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    return metrics

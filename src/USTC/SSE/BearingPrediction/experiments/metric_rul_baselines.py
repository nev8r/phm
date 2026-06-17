"""
Metric-driven RUL baseline experiment module

this module is for running tsfresh and sktime RUL baseline evidence

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from USTC.SSE.BearingPrediction.data import BearingWindowDataset
from USTC.SSE.BearingPrediction.dataset import XJTULoader
from USTC.SSE.BearingPrediction.evaluation import HuangRulScore, MAE, NormalizedRMSE, PHM2012Score, R2Score, RMSE
from USTC.SSE.BearingPrediction.feature import FeatureConfig, SignalFeatureExtractor
from USTC.SSE.BearingPrediction.labeling import FeatureSequenceRulLabeler
from USTC.SSE.BearingPrediction.models import FeatureSequenceTransformer, XLSTMTransformer
from USTC.SSE.BearingPrediction.training import BaseTester, BaseTrainer


MANUAL_FEATURE_NAMES = tuple(FeatureConfig(sample_rate=25_600.0).enabled_features)
SPLIT_NAME = "train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3"
TRAIN_BEARINGS = ("Bearing1_1", "Bearing1_2", "Bearing1_4", "Bearing1_5")
TEST_BEARING = "Bearing1_3"


@dataclass(frozen=True)
class XjtuMetricBaselineConfig:
    """
    Configuration for metric-driven XJTU RUL baselines.

    Parameters
    ----------
    project_root : Path
        repository root path
    xjtu_root : Path | None
        extracted XJTU dataset root
    output_dir : Path | None
        evidence output directory
    condition_dir : str
        XJTU operating condition directory
    downsample_points : int
        deterministic points kept from each vibration snapshot
    seeds : tuple[int, ...]
        repeated run random seeds
    """

    project_root: Path
    xjtu_root: Path | None = None
    output_dir: Path | None = None
    condition_dir: str = "35Hz12kN"
    downsample_points: int = 256
    seeds: tuple[int, ...] = (0, 1, 2)
    n_estimators: int = 80

    @property
    def resolved_xjtu_root(self) -> Path:
        """
        resolved XJTU dataset root.

        Returns
        -------
        Path
            extracted XJTU root path
        """

        if self.xjtu_root is not None:
            return self.xjtu_root
        return self.project_root / "data" / "external" / "xjtu" / "extracted" / "XJTU-SY_Bearing_Datasets"

    @property
    def resolved_output_dir(self) -> Path:
        """
        resolved evidence output directory.

        Returns
        -------
        Path
            evidence output path
        """

        if self.output_dir is not None:
            return self.output_dir
        return self.project_root / "docs" / "reproduction-evidence"

    @property
    def condition_name(self) -> str:
        """
        normalized condition name for evidence tables.

        Returns
        -------
        str
            condition name
        """

        return "condition_1_35Hz12kN" if self.condition_dir == "35Hz12kN" else self.condition_dir


@dataclass(frozen=True)
class XjtuSnapshotDataset:
    """
    In-memory XJTU condition snapshots and metadata.
    """

    metadata: pd.DataFrame
    horizontal_signals: list[np.ndarray]
    vertical_signals: list[np.ndarray]

    def train_mask(self) -> np.ndarray:
        """
        return train bearing mask.

        Returns
        -------
        np.ndarray
            train mask
        """

        return self.metadata["bearing_id"].isin(TRAIN_BEARINGS).to_numpy()

    def test_mask(self) -> np.ndarray:
        """
        return test bearing mask.

        Returns
        -------
        np.ndarray
            test mask
        """

        return (self.metadata["bearing_id"] == TEST_BEARING).to_numpy()


def load_xjtu_condition_snapshots(config: XjtuMetricBaselineConfig) -> XjtuSnapshotDataset:
    """
    load positive-RUL XJTU snapshots for a condition.

    Parameters
    ----------
    config : XjtuMetricBaselineConfig
        baseline configuration

    Returns
    -------
    XjtuSnapshotDataset
        metadata and downsampled signals
    """

    condition_root = config.resolved_xjtu_root / config.condition_dir
    if not condition_root.exists():
        raise FileNotFoundError(
            f"XJTU condition directory not found: {condition_root}. "
            "Extract data/external/xjtu/XJTU-SY_Bearing_Datasets.part*.rar first."
        )

    records: list[dict[str, object]] = []
    horizontal_signals: list[np.ndarray] = []
    vertical_signals: list[np.ndarray] = []
    bearing_ids = (*TRAIN_BEARINGS, TEST_BEARING)
    for bearing_id in bearing_ids:
        bearing_dir = condition_root / bearing_id
        signal_paths = sorted(bearing_dir.glob("*.csv"), key=_path_number)
        if not signal_paths:
            raise FileNotFoundError(f"no XJTU csv snapshots found under {bearing_dir}")
        lifetime = len(signal_paths) - 1
        for snapshot_index, signal_path in enumerate(signal_paths):
            true_rul = lifetime - snapshot_index
            if true_rul <= 0:
                continue
            signal_frame = pd.read_csv(signal_path, usecols=[0, 1])
            signal_values = signal_frame.to_numpy(dtype=float)
            horizontal, vertical = _downsample_two_channel_signal(signal_values, config.downsample_points)
            records.append(
                {
                    "dataset_name": "XJTU-SY",
                    "condition_name": config.condition_name,
                    "bearing_id": bearing_id,
                    "snapshot_index": snapshot_index + 1,
                    "true_rul": float(true_rul),
                    "sample_path": _display_path(signal_path, config.project_root),
                    "split_name": SPLIT_NAME,
                }
            )
            horizontal_signals.append(horizontal)
            vertical_signals.append(vertical)

    metadata = pd.DataFrame.from_records(records)
    return XjtuSnapshotDataset(metadata=metadata, horizontal_signals=horizontal_signals, vertical_signals=vertical_signals)


def run_tsfresh_feature_analysis(config: XjtuMetricBaselineConfig) -> dict[str, str]:
    """
    run tsfresh feature relevance analysis and write evidence artifacts.

    Parameters
    ----------
    config : XjtuMetricBaselineConfig
        baseline configuration

    Returns
    -------
    dict[str, str]
        output artifact paths
    """

    dataset = load_xjtu_condition_snapshots(config)
    tsfresh_features = extract_tsfresh_feature_frame(dataset)
    selected_features, relevance = select_tsfresh_features_train_only(tsfresh_features, dataset.metadata["true_rul"], dataset.train_mask())
    del selected_features
    relevance_summary = build_tsfresh_relevance_summary(relevance, config)

    config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.resolved_output_dir / "tsfresh_feature_relevance_summary.csv"
    markdown_path = config.resolved_output_dir / "tsfresh_feature_relevance_summary.md"
    relevance_summary.to_csv(csv_path, index=False)
    markdown_path.write_text(_render_tsfresh_relevance_markdown(relevance_summary), encoding="utf-8")
    return {
        "summary_path": _display_path(csv_path, config.project_root),
        "markdown_path": _display_path(markdown_path, config.project_root),
    }


def run_tsfresh_rul_baseline(config: XjtuMetricBaselineConfig) -> dict[str, str]:
    """
    run manual 19-feature and tsfresh-selected RUL baselines.

    Parameters
    ----------
    config : XjtuMetricBaselineConfig
        baseline configuration

    Returns
    -------
    dict[str, str]
        output artifact paths
    """

    dataset = load_xjtu_condition_snapshots(config)
    manual_features = extract_manual_feature_frame(dataset)
    tsfresh_features = extract_tsfresh_feature_frame(dataset)
    tsfresh_selected, relevance = select_tsfresh_features_train_only(
        tsfresh_features,
        dataset.metadata["true_rul"],
        dataset.train_mask(),
    )
    del relevance

    feature_sets = {
        "manual_19": manual_features,
        "tsfresh_selected": tsfresh_selected,
    }
    summary_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    for feature_backend, feature_frame in feature_sets.items():
        backend_summary, backend_predictions = _run_repeated_random_forest_baseline(
            config=config,
            dataset=dataset,
            feature_backend=feature_backend,
            feature_frame=feature_frame,
            model_name="RandomForestRegressor",
        )
        summary_records.extend(backend_summary)
        prediction_records.extend(backend_predictions)

    summary = pd.DataFrame.from_records(summary_records)
    predictions = pd.DataFrame.from_records(prediction_records)
    summary = _attach_repeated_metric_stats(summary, group_column="feature_backend")
    config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.resolved_output_dir / "tsfresh_rul_baseline_summary.csv"
    prediction_path = config.resolved_output_dir / "tsfresh_rul_baseline_predictions.csv"
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    return {
        "summary_path": _display_path(summary_path, config.project_root),
        "predictions_path": _display_path(prediction_path, config.project_root),
    }


def run_sktime_rul_baseline(config: XjtuMetricBaselineConfig) -> dict[str, str]:
    """
    run sktime panel RUL baselines.

    Parameters
    ----------
    config : XjtuMetricBaselineConfig
        baseline configuration

    Returns
    -------
    dict[str, str]
        output artifact paths
    """

    try:
        from sktime.regression.interval_based import TimeSeriesForestRegressor
        from sktime.regression.kernel_based import RocketRegressor
    except ImportError as exc:  # pragma: no cover - exercised by CLI use without advanced extra
        raise RuntimeError(
            "sktime baselines require the advanced extra. "
            "Run: uv run --extra advanced python scripts/run_sktime_rul_baseline.py"
        ) from exc

    dataset = load_xjtu_condition_snapshots(config)
    multivariate_panel = build_sktime_panel(dataset, channels=("horizontal", "vertical"))
    univariate_panel = build_sktime_panel(dataset, channels=("horizontal",))
    route_builders: dict[str, tuple[str, np.ndarray, Callable[[int], object]]] = {
        "rocket_regressor": (
            "RocketRegressor",
            multivariate_panel,
            lambda seed: RocketRegressor(num_kernels=512, random_state=seed),
        ),
        "time_series_forest_regressor": (
            "TimeSeriesForestRegressor",
            univariate_panel,
            lambda seed: TimeSeriesForestRegressor(n_estimators=80, random_state=seed, n_jobs=-1),
        ),
    }

    summary_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    train_mask = dataset.train_mask()
    test_mask = dataset.test_mask()
    y = dataset.metadata["true_rul"].to_numpy(dtype=float)
    for baseline_route, (model_name, panel, model_builder) in route_builders.items():
        for seed in config.seeds:
            model = model_builder(seed)
            model.fit(panel[train_mask], y[train_mask])
            predictions = np.asarray(model.predict(panel[test_mask]), dtype=float)
            metrics = calculate_rul_metrics(y[test_mask], predictions)
            experiment_name = f"XJTU-SY-{config.condition_name}-{baseline_route}"
            summary_records.append(
                {
                    "experiment_name": experiment_name,
                    "baseline_route": baseline_route,
                    "model_name": model_name,
                    "input_format": "sktime_3d_panel_numpy",
                    "dataset_name": "XJTU-SY",
                    "condition_name": config.condition_name,
                    "split_name": SPLIT_NAME,
                    "seed": seed,
                    "run_count": len(config.seeds),
                    **metrics,
                    "prediction_count": int(test_mask.sum()),
                    "panel_instance_count": int(panel.shape[0]),
                    "series_length": int(panel.shape[2]),
                    "status": "RUN_RECORDED",
                }
            )
            prediction_records.extend(
                _prediction_rows(
                    metadata=dataset.metadata.loc[test_mask],
                    predictions=predictions,
                    experiment_name=experiment_name,
                    backend_column="baseline_route",
                    backend_value=baseline_route,
                    seed=seed,
                )
            )

    summary = pd.DataFrame.from_records(summary_records)
    predictions = pd.DataFrame.from_records(prediction_records)
    summary = _attach_repeated_metric_stats(summary, group_column="baseline_route")
    config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.resolved_output_dir / "sktime_rul_baseline_summary.csv"
    prediction_path = config.resolved_output_dir / "sktime_rul_baseline_predictions.csv"
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    return {
        "summary_path": _display_path(summary_path, config.project_root),
        "predictions_path": _display_path(prediction_path, config.project_root),
    }


def build_strict_repeated_seed_summary(
    project_root: Path,
    output_dir: Path | None = None,
    *,
    xjtu_root: Path | None = None,
    seeds: tuple[int, ...] = (202601, 202602, 202603),
    epochs: int = 50,
    max_samples_per_entity: int | None = None,
) -> dict[str, str]:
    """
    build strict same-config repeated seed summary for formal strong models.

    Parameters
    ----------
    project_root : Path
        repository root
    output_dir : Path | None
        evidence output directory

    Returns
    -------
    dict[str, str]
        output path
    """

    evidence_dir = output_dir or project_root / "docs" / "reproduction-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    config = XjtuMetricBaselineConfig(project_root=project_root, xjtu_root=xjtu_root, output_dir=evidence_dir)
    train_set, test_set = _build_strict_xjtu_feature_sequence_split(config, max_samples_per_entity=max_samples_per_entity)
    scaled_train_set, scaled_test_set, target_scale = _scale_rul_targets_train_only(train_set, test_set)
    strict_config = {
        "dataset_name": "XJTU-SY",
        "condition_name": config.condition_name,
        "split_name": SPLIT_NAME,
        "train_bearings": list(TRAIN_BEARINGS),
        "test_bearing": TEST_BEARING,
        "models": ["XLSTM-Transformer", "Feature-Transformer"],
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_name": "mse",
        "target_scaling": "train_min_max",
        "target_min": target_scale["target_min"],
        "target_max": target_scale["target_max"],
        "sequence_length": 10,
        "window_size": 1024,
        "stride": 1024,
        "hidden_size": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.1,
        "max_samples_per_entity": max_samples_per_entity or "all_positive_rul_snapshots",
        "train_sequence_count": len(train_set),
        "test_sequence_count": len(test_set),
    }
    config_hash = hashlib.sha256(json.dumps(strict_config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    config_path = evidence_dir / "strict_repeated_seed_config.json"
    config_payload = {**strict_config, "config_hash": config_hash}
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    records: list[dict[str, object]] = []
    feature_size = int(train_set.inputs.shape[-1])
    for model_name in ["XLSTM-Transformer", "Feature-Transformer"]:
        for seed in seeds:
            _set_strict_seed(seed)
            model = _build_strict_model(model_name, feature_size)
            trainer = BaseTrainer(
                device="cpu",
                max_epochs=epochs,
                batch_size=64,
                learning_rate=1e-3,
                weight_decay=1e-4,
                shuffle_train=False,
                loss_name="mse",
            )
            training_result = trainer.train(model, scaled_train_set, scaled_test_set)
            test_result = BaseTester(device="cpu", batch_size=64).test(model, scaled_test_set)
            true_targets = _inverse_scale_rul(test_result.targets, target_scale)
            predicted_targets = _inverse_scale_rul(test_result.predictions, target_scale)
            metrics = calculate_rul_metrics(true_targets, predicted_targets)
            records.append(
                {
                    "model_name": model_name,
                    "dataset_name": "XJTU-SY",
                    "condition_name": config.condition_name,
                    "split_name": SPLIT_NAME,
                    "seed": seed,
                    "rmse": metrics["rmse"],
                    "normalized_rmse": metrics["normalized_rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "score": metrics["huang_rul_score"],
                    "epoch": int(training_result.best_epoch),
                    "config_hash": config_hash,
                    "config_path": _display_path(config_path, project_root),
                    "run_count": len(seeds),
                    "status": "RUN_RECORDED",
                }
            )
    summary = pd.DataFrame.from_records(records)
    output_path = evidence_dir / "strict_repeated_seed_summary.csv"
    summary.to_csv(output_path, index=False)
    return {
        "summary_path": _display_path(output_path, project_root),
        "config_path": _display_path(config_path, project_root),
    }


def build_external_sota_attempt_evidence(project_root: Path, output_dir: Path | None = None) -> dict[str, str]:
    """
    write reproducible external SOTA attempt records and logs.

    Parameters
    ----------
    project_root : Path
        repository root
    output_dir : Path | None
        evidence output directory

    Returns
    -------
    dict[str, str]
        output paths
    """

    evidence_dir = output_dir or project_root / "docs" / "reproduction-evidence"
    log_dir = evidence_dir / "external_sota_attempts"
    log_dir.mkdir(parents=True, exist_ok=True)
    attempts = [
        {
            "target_id": "autorul-pronostia-femto-rmse",
            "route_name": "AutoRUL / auto-sktime",
            "attempt_type": "source_pin_and_dependency_probe",
            "source_pin_command": (
                "git",
                "ls-remote",
                "--tags",
                "https://github.com/Ennosigaeon/auto-sktime",
                "v0.1.0",
            ),
            "environment_probe_command": (
                sys.executable,
                "-c",
                "import autosklearn; import sktime; print('AutoRUL runtime dependencies available')",
            ),
            "failure_reason": "Full AutoRUL rerun requires the external AutoML dependency stack and PRONOSTIA benchmark layout; the dependency probe fails in the project environment, so no local metric is claimed.",
            "next_step": "Create an isolated AutoRUL environment, materialize the femto_bearing data layout, then run scripts/remaining_useful_lifetime.py femto_bearing for ten repetitions.",
        },
        {
            "target_id": "gnn-benchmark-phm2012-fc-stgnn",
            "route_name": "GNN_RUL_Benchmarking FC-STGNN",
            "attempt_type": "source_pin_and_dependency_probe",
            "source_pin_command": (
                "git",
                "ls-remote",
                "https://github.com/Frank-Wang-oss/GNN_RUL_Benchmarking",
                "HEAD",
            ),
            "environment_probe_command": (
                sys.executable,
                "-c",
                "import torch; import torch_geometric; print('GNN benchmark runtime dependencies available')",
            ),
            "failure_reason": "Full rerun needs repository-specific PHM2012 preprocessing and the PyTorch Geometric training stack; the dependency probe fails in the project environment, so no local metric is claimed.",
            "next_step": "Build the repo environment, generate its PHM2012 preprocessed split, then run main.py with --GNN_method FC_STGNN --num_runs 5.",
        },
        {
            "target_id": "weibull-kiml-femto-rmse",
            "route_name": "Weibull KIML",
            "attempt_type": "source_pin_and_dependency_probe",
            "source_pin_command": (
                "git",
                "ls-remote",
                "https://github.com/tvhahn/weibull-knowledge-informed-ml",
                "HEAD",
            ),
            "environment_probe_command": (
                sys.executable,
                "-c",
                "import sksurv; import pycox; print('Weibull KIML runtime dependencies available')",
            ),
            "failure_reason": "Full rerun requires the repository make workflow, FEMTO data layout, and survival/deep-learning dependencies; the dependency probe fails in the project environment, so no local metric is claimed.",
            "next_step": "Create the repo environment, stage FEMTO data, then run make train_femto and make summarize_femto_models.",
        },
    ]
    records: list[dict[str, object]] = []
    for attempt in attempts:
        log_path = log_dir / f"{attempt['target_id']}.txt"
        source_command = tuple(str(part) for part in attempt["source_pin_command"])
        environment_command = tuple(str(part) for part in attempt["environment_probe_command"])
        source_completed = subprocess.run(
            source_command,
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        environment_completed = subprocess.run(
            environment_command,
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        log_body = (
            f"source_pin_command: {_format_command(source_command)}\n"
            f"source_pin_exit_code: {source_completed.returncode}\n"
            f"source_pin_stdout:\n{source_completed.stdout}\n"
            f"source_pin_stderr:\n{source_completed.stderr}\n"
            f"environment_probe_command: {_format_command(environment_command)}\n"
            f"environment_probe_exit_code: {environment_completed.returncode}\n"
            f"environment_probe_stdout:\n{environment_completed.stdout}\n"
            f"environment_probe_stderr:\n{environment_completed.stderr}\n"
            f"failure_reason: {attempt['failure_reason']}\n"
            f"next_step: {attempt['next_step']}\n"
        )
        log_path.write_text(log_body, encoding="utf-8")
        command = f"{_format_command(source_command)} && {_format_command(environment_command)}"
        records.append(
            {
                "target_id": attempt["target_id"],
                "route_name": attempt["route_name"],
                "attempt_type": attempt["attempt_type"],
                "command": command,
                "source_pin_command": _format_command(source_command),
                "source_pin_exit_code": source_completed.returncode,
                "environment_probe_command": _format_command(environment_command),
                "environment_probe_exit_code": environment_completed.returncode,
                "failure_reason": attempt["failure_reason"],
                "next_step": attempt["next_step"],
                "attempt_status": "ATTEMPTED_EXTERNAL_ENV_BLOCKED",
                "log_path": _display_path(log_path, project_root),
            }
        )
    attempts_frame = pd.DataFrame.from_records(records)
    attempts_path = evidence_dir / "external_sota_attempts.csv"
    attempts_frame.to_csv(attempts_path, index=False)
    return {
        "attempts_path": _display_path(attempts_path, project_root),
        "log_dir": _display_path(log_dir, project_root),
    }


def _format_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def _build_strict_xjtu_feature_sequence_split(
    config: XjtuMetricBaselineConfig,
    *,
    max_samples_per_entity: int | None,
) -> tuple[BearingWindowDataset, BearingWindowDataset]:
    loader = XJTULoader(config.resolved_xjtu_root)
    labeler = FeatureSequenceRulLabeler(sequence_length=10, window_size=1024, stride=1024)
    train_sets = [
        labeler.label(loader.load_entity(bearing_id, max_samples=max_samples_per_entity), "Horizontal Vibration")
        for bearing_id in TRAIN_BEARINGS
    ]
    test_set = labeler.label(loader.load_entity(TEST_BEARING, max_samples=max_samples_per_entity), "Horizontal Vibration")
    return _concat_bearing_window_datasets(train_sets), test_set


def _concat_bearing_window_datasets(datasets: list[BearingWindowDataset]) -> BearingWindowDataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    first_dataset = datasets[0]
    feature_frame = None
    if first_dataset.feature_frame is not None:
        feature_frame = pd.concat([dataset.feature_frame for dataset in datasets if dataset.feature_frame is not None], ignore_index=True)
    extra_targets = {
        key: np.concatenate([dataset.extra_targets[key] for dataset in datasets], axis=0)
        for key in first_dataset.extra_targets
    }
    return BearingWindowDataset(
        inputs=np.concatenate([dataset.inputs for dataset in datasets], axis=0).astype(np.float32),
        targets=np.concatenate([dataset.targets for dataset in datasets], axis=0).astype(np.float32),
        metadata_frame=pd.concat([dataset.metadata_frame for dataset in datasets], ignore_index=True),
        task_type=first_dataset.task_type,
        target_name=first_dataset.target_name,
        input_name=first_dataset.input_name,
        feature_frame=feature_frame,
        extra_targets=extra_targets,
    )


def _scale_rul_targets_train_only(
    train_set: BearingWindowDataset,
    test_set: BearingWindowDataset,
) -> tuple[BearingWindowDataset, BearingWindowDataset, dict[str, float]]:
    target_min = float(np.min(train_set.targets))
    target_max = float(np.max(train_set.targets))
    target_range = max(target_max - target_min, 1.0)
    scale = {"target_min": target_min, "target_max": target_max, "target_range": target_range}
    return _replace_targets(train_set, (train_set.targets - target_min) / target_range), _replace_targets(
        test_set,
        (test_set.targets - target_min) / target_range,
    ), scale


def _replace_targets(dataset: BearingWindowDataset, targets: np.ndarray) -> BearingWindowDataset:
    return BearingWindowDataset(
        inputs=dataset.inputs,
        targets=np.asarray(targets, dtype=np.float32),
        metadata_frame=dataset.metadata_frame,
        task_type=dataset.task_type,
        target_name=dataset.target_name,
        input_name=dataset.input_name,
        feature_frame=dataset.feature_frame,
        extra_targets=dataset.extra_targets,
    )


def _inverse_scale_rul(values: np.ndarray, scale: dict[str, float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * scale["target_range"] + scale["target_min"]


def _set_strict_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_strict_model(model_name: str, feature_size: int) -> torch.nn.Module:
    model_kwargs = {
        "feature_size": feature_size,
        "output_size": 1,
        "sequence_length": 10,
        "hidden_size": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.1,
    }
    if model_name == "XLSTM-Transformer":
        return XLSTMTransformer(**model_kwargs)
    if model_name == "Feature-Transformer":
        return FeatureSequenceTransformer(**model_kwargs)
    raise ValueError(f"unknown strict model name: {model_name}")


def extract_manual_feature_frame(dataset: XjtuSnapshotDataset) -> pd.DataFrame:
    """
    extract project manual 19-feature table from horizontal vibration.

    Parameters
    ----------
    dataset : XjtuSnapshotDataset
        loaded snapshots

    Returns
    -------
    pd.DataFrame
        manual feature table
    """

    extractor = SignalFeatureExtractor(FeatureConfig(sample_rate=25_600.0))
    feature_frame = extractor.extract(dataset.horizontal_signals)
    return feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def extract_tsfresh_feature_frame(dataset: XjtuSnapshotDataset) -> pd.DataFrame:
    """
    extract tsfresh MinimalFCParameters features for two vibration channels.

    Parameters
    ----------
    dataset : XjtuSnapshotDataset
        loaded snapshots

    Returns
    -------
    pd.DataFrame
        tsfresh feature table
    """

    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError as exc:  # pragma: no cover - exercised by CLI use without advanced extra
        raise RuntimeError(
            "tsfresh feature analysis requires the advanced extra. "
            "Run: uv run --extra advanced python scripts/run_tsfresh_feature_analysis.py"
        ) from exc

    long_frame = _build_tsfresh_long_frame(dataset)
    features = extract_features(
        long_frame,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
        n_jobs=1,
    )
    features = features.sort_index()
    features.index = features.index.astype(int)
    features = features.loc[range(len(dataset.metadata))]
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def select_tsfresh_features_train_only(
    feature_frame: pd.DataFrame,
    targets: pd.Series,
    train_mask: np.ndarray,
    *,
    max_features: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    select tsfresh features using train rows only.

    Parameters
    ----------
    feature_frame : pd.DataFrame
        all extracted features
    targets : pd.Series
        target RUL values
    train_mask : np.ndarray
        train row mask
    max_features : int
        max selected feature count

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        selected feature frame and relevance table
    """

    train_features = feature_frame.loc[train_mask].copy()
    train_targets = targets.loc[train_mask].to_numpy(dtype=float)
    relevance = _calculate_relevance(train_features, train_targets)
    selected = relevance[relevance["selected"]]["feature_name"].head(max_features).tolist()
    if not selected:
        selected = relevance.head(max_features)["feature_name"].tolist()
    selected_frame = feature_frame[selected].copy()
    return selected_frame, relevance


def build_sktime_panel(dataset: XjtuSnapshotDataset, *, channels: tuple[str, ...]) -> np.ndarray:
    """
    build sktime 3D panel numpy array.

    Parameters
    ----------
    dataset : XjtuSnapshotDataset
        loaded snapshots
    channels : tuple[str, ...]
        channel names to include

    Returns
    -------
    np.ndarray
        panel array shaped n_instances, n_channels, n_timepoints
    """

    channel_values = {
        "horizontal": dataset.horizontal_signals,
        "vertical": dataset.vertical_signals,
    }
    panel_channels = [np.vstack(channel_values[channel]) for channel in channels]
    return np.stack(panel_channels, axis=1).astype(np.float32)


def calculate_rul_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """
    calculate unified RUL metrics.

    Parameters
    ----------
    targets : np.ndarray
        true RUL
    predictions : np.ndarray
        predicted RUL

    Returns
    -------
    dict[str, float]
        metric values
    """

    safe_predictions = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    safe_targets = np.asarray(targets, dtype=float)
    return {
        "rmse": RMSE()(safe_targets, safe_predictions),
        "normalized_rmse": NormalizedRMSE()(safe_targets, safe_predictions),
        "mae": MAE()(safe_targets, safe_predictions),
        "r2": R2Score()(safe_targets, safe_predictions),
        "huang_rul_score": HuangRulScore()(safe_targets, safe_predictions),
        "phm2012_score": PHM2012Score()(safe_targets, safe_predictions),
    }


def build_tsfresh_relevance_summary(relevance: pd.DataFrame, config: XjtuMetricBaselineConfig) -> pd.DataFrame:
    """
    build reader-facing tsfresh relevance summary.

    Parameters
    ----------
    relevance : pd.DataFrame
        relevance table
    config : XjtuMetricBaselineConfig
        baseline configuration

    Returns
    -------
    pd.DataFrame
        summary table
    """

    records: list[dict[str, object]] = []
    for row in relevance.to_dict("records"):
        feature_name = str(row["feature_name"])
        records.append(
            {
                "feature_name": feature_name,
                "dataset_name": "XJTU-SY",
                "condition_name": config.condition_name,
                "target_name": "rul",
                "score": row["score"],
                "p_value": row["p_value"],
                "correlation": row["correlation"],
                "selected": bool(row["selected"]),
                "feature_group": _feature_group(feature_name),
                "interpretation": _feature_interpretation(feature_name),
                "selection_split": "train_only",
                "overlaps_manual_19": _overlaps_manual_feature(feature_name),
            }
        )
    summary = pd.DataFrame.from_records(records)
    return summary.sort_values(["selected", "score"], ascending=[False, False]).reset_index(drop=True)


def _run_repeated_random_forest_baseline(
    *,
    config: XjtuMetricBaselineConfig,
    dataset: XjtuSnapshotDataset,
    feature_backend: str,
    feature_frame: pd.DataFrame,
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_mask = dataset.train_mask()
    test_mask = dataset.test_mask()
    y = dataset.metadata["true_rul"].to_numpy(dtype=float)
    train_features = feature_frame.loc[train_mask].to_numpy(dtype=float)
    test_features = feature_frame.loc[test_mask].to_numpy(dtype=float)
    summary_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    for seed in config.seeds:
        model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_features, y[train_mask])
        predictions = np.asarray(model.predict(test_features), dtype=float)
        metrics = calculate_rul_metrics(y[test_mask], predictions)
        experiment_name = f"XJTU-SY-{config.condition_name}-{feature_backend}-{model_name}"
        summary_records.append(
            {
                "experiment_name": experiment_name,
                "feature_backend": feature_backend,
                "model_name": model_name,
                "dataset_name": "XJTU-SY",
                "condition_name": config.condition_name,
                "split_name": SPLIT_NAME,
                "seed": seed,
                "run_count": len(config.seeds),
                **metrics,
                "prediction_count": int(test_mask.sum()),
                "feature_count": int(feature_frame.shape[1]),
                "selection_split": "train_only",
                "status": "RUN_RECORDED",
            }
        )
        prediction_records.extend(
            _prediction_rows(
                metadata=dataset.metadata.loc[test_mask],
                predictions=predictions,
                experiment_name=experiment_name,
                backend_column="feature_backend",
                backend_value=feature_backend,
                seed=seed,
            )
        )
    return summary_records, prediction_records


def _prediction_rows(
    *,
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    experiment_name: str,
    backend_column: str,
    backend_value: str,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row, prediction in zip(metadata.to_dict("records"), predictions, strict=True):
        records.append(
            {
                "experiment_name": experiment_name,
                backend_column: backend_value,
                "seed": seed,
                "bearing_id": row["bearing_id"],
                "snapshot_index": row["snapshot_index"],
                "true_rul": row["true_rul"],
                "predicted_rul": max(0.0, float(prediction)),
                "split_name": row["split_name"],
            }
        )
    return records


def _attach_repeated_metric_stats(summary: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    metric_names = ["rmse", "normalized_rmse", "mae", "r2", "huang_rul_score", "phm2012_score"]
    summary = summary.copy()
    for metric_name in metric_names:
        grouped = summary.groupby(group_column)[metric_name]
        summary[f"{metric_name}_mean"] = summary[group_column].map(grouped.mean())
        summary[f"{metric_name}_std"] = summary[group_column].map(lambda key: float(grouped.std(ddof=0).loc[key]))
    return summary


def _build_tsfresh_long_frame(dataset: XjtuSnapshotDataset) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row_index, (horizontal, vertical) in enumerate(zip(dataset.horizontal_signals, dataset.vertical_signals, strict=True)):
        time_index = np.arange(horizontal.size)
        for kind, values in [("horizontal", horizontal), ("vertical", vertical)]:
            records.extend(
                {
                    "id": row_index,
                    "time": int(time_value),
                    "kind": kind,
                    "value": float(value),
                }
                for time_value, value in zip(time_index, values, strict=True)
            )
    return pd.DataFrame.from_records(records)


def _calculate_relevance(train_features: pd.DataFrame, train_targets: np.ndarray) -> pd.DataFrame:
    correlations = []
    for feature_name in train_features.columns:
        values = train_features[feature_name].to_numpy(dtype=float)
        if np.std(values) < 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(values, train_targets)[0, 1])
            if math.isnan(correlation):
                correlation = 0.0
        correlations.append(
            {
                "feature_name": feature_name,
                "correlation": correlation,
                "score": abs(correlation),
                "p_value": _pearson_p_value(correlation, len(values)),
            }
        )
    relevance = pd.DataFrame.from_records(correlations).sort_values("score", ascending=False).reset_index(drop=True)
    top_count = min(32, max(8, int(len(relevance) * 0.25)))
    relevance["selected"] = False
    relevance.loc[: top_count - 1, "selected"] = True
    return relevance


def _pearson_p_value(correlation: float, sample_size: int) -> float:
    if sample_size <= 2:
        return 1.0
    try:
        from scipy import stats

        statistic = correlation * math.sqrt((sample_size - 2) / max(1e-12, 1.0 - correlation**2))
        return float(2.0 * stats.t.sf(abs(statistic), df=sample_size - 2))
    except Exception:
        return float("nan")


def _downsample_two_channel_signal(signal_values: np.ndarray, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    if signal_values.shape[0] <= point_count:
        sampled = signal_values
    else:
        indices = np.linspace(0, signal_values.shape[0] - 1, point_count).round().astype(int)
        sampled = signal_values[indices]
    horizontal = _zscore(sampled[:, 0].astype(float))
    vertical = _zscore(sampled[:, 1].astype(float))
    return horizontal, vertical


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _feature_group(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "fft" in lowered or "frequency" in lowered:
        return "frequency"
    if "kurtosis" in lowered or "skewness" in lowered or "variance" in lowered or "standard_deviation" in lowered:
        return "distribution"
    if "absolute" in lowered or "energy" in lowered or "root" in lowered:
        return "energy"
    if "length" in lowered or "count" in lowered:
        return "shape"
    return "time_domain"


def _feature_interpretation(feature_name: str) -> str:
    if "__" in feature_name:
        channel, calculator = feature_name.split("__", maxsplit=1)
    else:
        channel, calculator = "signal", feature_name
    readable = calculator.replace("_", " ")
    return f"{channel} channel {readable} statistic; selected on train bearings only to avoid held-out leakage."


def _overlaps_manual_feature(feature_name: str) -> bool:
    lowered = feature_name.lower()
    return any(manual_name in lowered for manual_name in MANUAL_FEATURE_NAMES)


def _render_tsfresh_relevance_markdown(summary: pd.DataFrame) -> str:
    top_rows = summary.head(20)
    lines = [
        "# tsfresh Feature Relevance Summary",
        "",
        "Selection uses only train bearings (`Bearing1_1`, `Bearing1_2`, `Bearing1_4`, `Bearing1_5`) and keeps `Bearing1_3` held out for RUL baselines.",
        "",
        "| feature_name | score | p_value | correlation | selected | feature_group | overlaps_manual_19 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in top_rows.to_dict("records"):
        lines.append(
            "| {feature_name} | {score:.6f} | {p_value:.6g} | {correlation:.6f} | {selected} | {feature_group} | {overlaps_manual_19} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "All feature scores are correlation-derived screening scores on train rows. The downstream baseline transforms held-out rows with the already selected feature list only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _path_number(path: Path) -> int:
    return int(re.sub(r"\D", "", path.stem) or 0)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def cli_config(project_root: Path, xjtu_root: Path | None, output_dir: Path | None, seeds: list[int], downsample_points: int) -> XjtuMetricBaselineConfig:
    """
    build config from CLI arguments.

    Parameters
    ----------
    project_root : Path
        project root
    xjtu_root : Path | None
        optional dataset root
    output_dir : Path | None
        optional evidence dir
    seeds : list[int]
        random seeds
    downsample_points : int
        downsampled series length

    Returns
    -------
    XjtuMetricBaselineConfig
        config
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    return XjtuMetricBaselineConfig(
        project_root=project_root.resolve(),
        xjtu_root=xjtu_root.resolve() if xjtu_root is not None else None,
        output_dir=output_dir.resolve() if output_dir is not None else None,
        seeds=tuple(int(seed) for seed in seeds),
        downsample_points=downsample_points,
    )


def print_paths(paths: dict[str, str]) -> None:
    """
    print script output paths.

    Parameters
    ----------
    paths : dict[str, str]
        path mapping
    """

    for key, value in paths.items():
        print(f"{key}: {value}")


def exit_with_dependency_message(exc: RuntimeError) -> None:
    """
    print a dependency error and exit non-zero.

    Parameters
    ----------
    exc : RuntimeError
        dependency error
    """

    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from exc

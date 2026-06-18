"""
Training configuration module

this file is for loading and normalizing bearing PHM training YAML files

created by zy

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


TRAINING_OVERRIDE_KEYS = {"epochs", "sequence_length", "batch_size", "learning_rate", "weight_decay"}
DEFAULT_CONFIG_ROOT = Path("configs/training")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dataset_for_task(task: str) -> str:
    return "PHM2012" if task == "rul" else "XJTU-SY"


def _model_for_task(task: str) -> str:
    return "CBAM-CNN-LSTM" if task == "rul" else "ResCNN-LSTM"


def _split_preview(task: str, sample: bool) -> dict[str, str]:
    if task == "rul":
        return {
            "strategy": "bearing_holdout_with_train_validation_split",
            "train": "sample 前 N-1 个轴承" if sample else "PHM2012 Learning_set 训练轴承",
            "val": "训练轴承窗口随机 15%",
            "test": "sample 最后 1 个轴承" if sample else "论文主线测试轴承 Bearing1_3/1_4/1_5/1_6/1_7",
        }
    return {
        "strategy": "random_window_split",
        "train": "全体窗口随机 70%",
        "val": "全体窗口随机 15%",
        "test": "全体窗口随机 15%",
    }


def load_training_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"training config must be a YAML mapping: {config_path}")
    return data


def list_training_config_files(config_root: Path | str = DEFAULT_CONFIG_ROOT) -> list[Path]:
    root = Path(config_root)
    if not root.exists():
        return []
    files = [*root.glob("*.yaml"), *root.glob("*.yml")]
    return sorted(path for path in files if path.is_file())


def _mode_to_sample(mode: Any, default: bool) -> bool:
    if mode is None:
        return default
    normalized = str(mode).strip().lower()
    if normalized in {"sample", "smoke", "quick"}:
        return True
    if normalized in {"full", "paper"}:
        return False
    raise ValueError(f"unsupported data mode in training config: {mode}")


def resolve_training_config(
    raw_config: dict[str, Any] | None = None,
    *,
    config_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw_config or {}
    cli = {key: value for key, value in (cli_overrides or {}).items() if value is not None}
    data = _as_dict(raw.get("data"))
    trainer = _as_dict(raw.get("trainer"))
    training = _as_dict(raw.get("training"))
    model = _as_dict(raw.get("model"))

    task = cli.get("task") or raw.get("task") or data.get("task")
    if task not in {"rul", "fault"}:
        raise ValueError("training config must define task as 'rul' or 'fault'")

    preset = str(cli.get("preset") or raw.get("preset") or training.get("preset") or "paper")
    if preset not in {"paper", "smoke"}:
        raise ValueError("training preset must be 'paper' or 'smoke'")

    mode = cli.get("data_mode") or raw.get("mode") or data.get("mode")
    default_sample = True if not raw else False
    sample = _mode_to_sample(mode, default=bool(raw.get("sample", default_sample)))
    device = str(cli.get("device") or raw.get("device") or trainer.get("device") or "auto")
    seed = int(cli.get("seed") or raw.get("seed") or trainer.get("seed") or 42)
    output_dir = str(cli.get("output_dir") or raw.get("output_dir") or training.get("output_dir") or "outputs/runs")
    run_dir = cli.get("run_dir") or raw.get("run_dir")

    training_overrides = {
        key: training[key]
        for key in TRAINING_OVERRIDE_KEYS
        if key in training
    }
    architecture = _as_dict(model.get("architecture"))
    dataset_config = {
        "task": task,
        "dataset": str(data.get("dataset") or _dataset_for_task(task)),
        "mode": "sample" if sample else "full",
    }
    for key in ("root", "cache", "feature_set"):
        if key in data:
            dataset_config[key] = data[key]

    trainer_config = {
        "name": str(trainer.get("name") or "BaseTrainer"),
        "device": device,
        "seed": seed,
    }
    model_config = {
        "name": str(model.get("name") or _model_for_task(task)),
        "architecture": architecture,
    }
    return {
        "task": task,
        "preset": preset,
        "sample": sample,
        "device": device,
        "seed": seed,
        "output_dir": output_dir,
        "run_dir": str(run_dir) if run_dir else None,
        "source_config_path": str(config_path) if config_path else "",
        "dataset_config": dataset_config,
        "trainer_config": trainer_config,
        "training_overrides": training_overrides,
        "model_config": model_config,
        "architecture_overrides": architecture,
        "split_preview": _split_preview(task, sample),
    }

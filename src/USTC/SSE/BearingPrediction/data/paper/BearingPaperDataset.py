"""
Bearing paper dataset module

this file is for building paper reproduction datasets and feature caches

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from USTC.SSE.BearingPrediction.data.process.array.FFTMagnitudeProcessor import FFTMagnitudeProcessor
from USTC.SSE.BearingPrediction.data.process.array.FrequencyBandEnergyProcessor import FrequencyBandEnergyProcessor
from USTC.SSE.BearingPrediction.data.process.array.SpectralFeatureProcessor import SpectralFeatureProcessor
from USTC.SSE.BearingPrediction.data.process.array.TimeDomainFeatureProcessor import TimeDomainFeatureProcessor


DEFAULT_TIME_FEATURES = (
    "mean",
    "std",
    "rms",
    "mean_abs",
    "ptp",
    "skewness",
    "kurtosis",
    "crest_factor",
)
DEFAULT_SPECTRAL_FEATURES = (
    "centroid",
    "bandwidth",
    "rms_frequency",
    "peak_frequency",
    "entropy",
    "flatness",
    "rolloff",
)
DEFAULT_FREQUENCY_BANDS = (
    (0.0, 1000.0),
    (1000.0, 3000.0),
    (3000.0, 6000.0),
    (6000.0, 10000.0),
    (10000.0, 12800.0),
)
PHM2012_SAMPLING_RATE = 25600
XJTU_SAMPLING_RATE = 25600

PHM2012_LEARNING_BEARINGS = (
    "Bearing1_1",
    "Bearing1_2",
    "Bearing2_1",
    "Bearing2_2",
    "Bearing3_1",
    "Bearing3_2",
)
PHM2012_FULL_TEST_BEARINGS = (
    "Bearing1_3",
    "Bearing1_4",
    "Bearing1_5",
    "Bearing1_6",
    "Bearing1_7",
    "Bearing2_3",
    "Bearing2_4",
    "Bearing2_5",
    "Bearing2_6",
    "Bearing2_7",
    "Bearing3_3",
)
XJTU_FAULT_LABELS = {
    "Bearing1_1": ("OF",),
    "Bearing1_2": ("OF",),
    "Bearing1_3": ("OF",),
    "Bearing1_4": ("CF",),
    "Bearing1_5": ("IF", "OF"),
    "Bearing2_1": ("IF",),
    "Bearing2_2": ("OF",),
    "Bearing2_3": ("CF",),
    "Bearing2_4": ("OF",),
    "Bearing2_5": ("OF",),
    "Bearing3_1": ("OF",),
    "Bearing3_2": ("IF", "OF", "CF", "BF"),
    "Bearing3_3": ("IF",),
    "Bearing3_4": ("IF",),
    "Bearing3_5": ("OF",),
}
XJTU_FAULT_TYPES = ("OF", "IF", "CF", "BF")
XJTU_HEALTH_STATES = ("Healthy", "Faulty")


def numeric_sort_key(path: str | Path) -> tuple[int, str]:
    name = Path(path).name
    match = re.search(r"\d+", name)
    return (int(match.group()) if match else -1, name)


def read_phm2012_acc_file(file_path: str | Path, bearing_name: str) -> np.ndarray:
    sep = ";" if bearing_name == "Bearing1_4" else ","
    dataframe = pd.read_csv(file_path, header=None, sep=sep)
    return dataframe.iloc[:, -2:].to_numpy(dtype=np.float32)


def read_xjtu_csv_file(file_path: str | Path) -> np.ndarray:
    dataframe = pd.read_csv(file_path)
    columns = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
    if all(column in dataframe.columns for column in columns):
        return dataframe[columns].to_numpy(dtype=np.float32)
    return dataframe.iloc[:, -2:].to_numpy(dtype=np.float32)


def list_phm2012_acc_files(root: str | Path, split: str, bearing_name: str) -> list[Path]:
    bearing_dir = Path(root) / split / bearing_name
    return sorted(bearing_dir.glob("acc_*.csv"), key=numeric_sort_key)


def list_xjtu_csv_files(root: str | Path, condition: str, bearing_name: str) -> list[Path]:
    bearing_dir = Path(root) / condition / bearing_name
    return sorted(bearing_dir.glob("*.csv"), key=numeric_sort_key)


def extract_feature_vector(
    signal: np.ndarray,
    sampling_rate: int,
    fft_bins: int = 256,
    channels: Sequence[int] = (0,),
    frequency_bands: Sequence[tuple[float, float]] = DEFAULT_FREQUENCY_BANDS,
    include_handcrafted: bool = True,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    features = []
    for channel in channels:
        channel_signal = signal[:, channel]
        features.append(
            FFTMagnitudeProcessor(
                sampling_rate=sampling_rate,
                n_bins=fft_bins,
                include_dc=False,
                log_scale=True,
                window="hann",
            ).run(channel_signal)
        )
        if include_handcrafted:
            features.append(TimeDomainFeatureProcessor(DEFAULT_TIME_FEATURES).run(channel_signal))
            features.append(
                SpectralFeatureProcessor(
                    sampling_rate=sampling_rate,
                    features=DEFAULT_SPECTRAL_FEATURES,
                    include_dc=False,
                ).run(channel_signal)
            )
            features.append(
                FrequencyBandEnergyProcessor(
                    sampling_rate=sampling_rate,
                    bands=frequency_bands,
                    relative=True,
                    include_dc=False,
                ).run(channel_signal)
            )

    return np.concatenate(features).astype(np.float32)


def build_phm2012_rul_feature_cache(
    root: str | Path,
    cache_path: str | Path,
    fft_bins: int = 256,
    include_handcrafted: bool = True,
    force: bool = False,
    progress_interval: int | None = 500,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_feature_cache(cache_path)

    features = []
    targets = []
    file_indices = []
    split_names = []
    bearing_names = []
    ranges = {}
    processed = 0

    for split, bearings in (
        ("Learning_set", PHM2012_LEARNING_BEARINGS),
        ("Full_Test_Set", PHM2012_FULL_TEST_BEARINGS),
    ):
        for bearing_name in bearings:
            files = list_phm2012_acc_files(root, split, bearing_name)
            start = len(features)
            denominator = max(len(files) - 1, 1)
            for index, file_path in enumerate(files):
                signal = read_phm2012_acc_file(file_path, bearing_name)
                features.append(
                    extract_feature_vector(
                        signal,
                        sampling_rate=PHM2012_SAMPLING_RATE,
                        fft_bins=fft_bins,
                        channels=(0,),
                        include_handcrafted=include_handcrafted,
                    )
                )
                targets.append([1.0 - index / denominator])
                file_indices.append(index)
                split_names.append(split)
                bearing_names.append(bearing_name)
                processed += 1
                if progress_interval and processed % progress_interval == 0:
                    print(f"[PHM2012] processed {processed} acceleration files")
            ranges[bearing_name] = (start, len(features))

    metadata = {
        "dataset": "PHM2012",
        "task": "RUL",
        "sampling_rate": PHM2012_SAMPLING_RATE,
        "fft_bins": fft_bins,
        "include_handcrafted": include_handcrafted,
        "feature_count": int(np.asarray(features).shape[1]),
        "ranges": ranges,
    }
    result = {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "file_indices": np.asarray(file_indices, dtype=np.int32),
        "splits": np.asarray(split_names),
        "bearing_names": np.asarray(bearing_names),
        "ranges": ranges,
        "metadata": metadata,
    }
    save_feature_cache(cache_path, result)
    return result


def build_xjtu_fault_feature_cache(
    root: str | Path,
    cache_path: str | Path,
    fft_bins: int = 256,
    include_handcrafted: bool = True,
    force: bool = False,
    progress_interval: int | None = 500,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_feature_cache(cache_path)

    root = Path(root)
    features = []
    targets = []
    file_indices = []
    condition_names = []
    bearing_names = []
    ranges = {}
    processed = 0

    for condition_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        for bearing_dir in sorted([path for path in condition_dir.iterdir() if path.is_dir()], key=lambda p: p.name):
            bearing_name = bearing_dir.name
            files = list_xjtu_csv_files(root, condition_dir.name, bearing_name)
            start = len(features)
            label = _multi_hot_fault_label(XJTU_FAULT_LABELS[bearing_name])
            for index, file_path in enumerate(files):
                signal = read_xjtu_csv_file(file_path)
                features.append(
                    extract_feature_vector(
                        signal,
                        sampling_rate=XJTU_SAMPLING_RATE,
                        fft_bins=fft_bins,
                        channels=(0, 1),
                        include_handcrafted=include_handcrafted,
                    )
                )
                targets.append(label)
                file_indices.append(index)
                condition_names.append(condition_dir.name)
                bearing_names.append(bearing_name)
                processed += 1
                if progress_interval and processed % progress_interval == 0:
                    print(f"[XJTU-SY] processed {processed} csv files")
            ranges[bearing_name] = (start, len(features))

    metadata = {
        "dataset": "XJTU-SY",
        "task": "multi_label_fault_diagnosis",
        "sampling_rate": XJTU_SAMPLING_RATE,
        "fft_bins": fft_bins,
        "include_handcrafted": include_handcrafted,
        "fault_types": XJTU_FAULT_TYPES,
        "feature_count": int(np.asarray(features).shape[1]),
        "ranges": ranges,
    }
    result = {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "file_indices": np.asarray(file_indices, dtype=np.int32),
        "conditions": np.asarray(condition_names),
        "bearing_names": np.asarray(bearing_names),
        "ranges": ranges,
        "metadata": metadata,
    }
    save_feature_cache(cache_path, result)
    return result


def estimate_three_sigma_fot_index(
    values: Sequence[float] | np.ndarray,
    ratio: float = 3.0,
    max_consecution: int = 5,
    consecution_ratio: float = 0.3,
    healthy_ratio: float = 0.3,
    min_bound: float = 0.1,
    max_rms: float = 2.0,
) -> int:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("values must not be empty")

    healthy_size = max(int(values.size * healthy_ratio), 1)
    healthy_values = values[:healthy_size]
    threshold = float(healthy_values.mean() + ratio * healthy_values.std() + min_bound)
    consecution_max = min(max(int(consecution_ratio * values.size), 1), max_consecution)

    consecution_count = 0
    for index, value in enumerate(values):
        if value > threshold or value > max_rms:
            consecution_count += 1
            if consecution_count == consecution_max:
                return max(index - consecution_max + 1, 0)
            if index == values.size - 1:
                return max(index - consecution_count + 1, 0)
        else:
            consecution_count = 0
    return values.size


def build_xjtu_binary_fault_diagnosis_feature_cache(
    root: str | Path,
    cache_path: str | Path,
    fft_bins: int = 256,
    include_handcrafted: bool = True,
    force: bool = False,
    progress_interval: int | None = 500,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_feature_cache(cache_path)

    root = Path(root)
    features = []
    targets = []
    file_indices = []
    condition_names = []
    bearing_names = []
    ranges = {}
    fot_indices = {}
    processed = 0

    for condition_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        for bearing_dir in sorted([path for path in condition_dir.iterdir() if path.is_dir()], key=lambda p: p.name):
            bearing_name = bearing_dir.name
            files = list_xjtu_csv_files(root, condition_dir.name, bearing_name)
            start = len(features)
            bearing_features = []
            horizontal_rms = []
            for index, file_path in enumerate(files):
                signal = read_xjtu_csv_file(file_path)
                bearing_features.append(
                    extract_feature_vector(
                        signal,
                        sampling_rate=XJTU_SAMPLING_RATE,
                        fft_bins=fft_bins,
                        channels=(0, 1),
                        include_handcrafted=include_handcrafted,
                    )
                )
                horizontal_rms.append(float(np.sqrt(np.mean(np.square(signal[:, 0])))))
                file_indices.append(index)
                condition_names.append(condition_dir.name)
                bearing_names.append(bearing_name)
                processed += 1
                if progress_interval and processed % progress_interval == 0:
                    print(f"[XJTU-SY diagnosis] processed {processed} csv files")

            fot_index = estimate_three_sigma_fot_index(horizontal_rms)
            fot_indices[bearing_name] = int(fot_index)
            features.extend(bearing_features)
            targets.extend([[0 if index < fot_index else 1] for index in range(len(bearing_features))])
            ranges[bearing_name] = (start, len(features))

    metadata = {
        "dataset": "XJTU-SY",
        "task": "binary_fault_diagnosis",
        "sampling_rate": XJTU_SAMPLING_RATE,
        "fft_bins": fft_bins,
        "include_handcrafted": include_handcrafted,
        "target_names": XJTU_HEALTH_STATES,
        "feature_count": int(np.asarray(features).shape[1]),
        "ranges": ranges,
        "fot_indices": fot_indices,
    }
    result = {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.int64),
        "file_indices": np.asarray(file_indices, dtype=np.int32),
        "conditions": np.asarray(condition_names),
        "bearing_names": np.asarray(bearing_names),
        "ranges": ranges,
        "metadata": metadata,
    }
    save_feature_cache(cache_path, result)
    return result


def save_feature_cache(cache_path: Path, data: Mapping) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=data["features"],
        targets=data["targets"],
        file_indices=data["file_indices"],
        splits=data.get("splits", np.asarray([])),
        conditions=data.get("conditions", np.asarray([])),
        bearing_names=data["bearing_names"],
        metadata=json.dumps(data["metadata"]),
    )


def load_feature_cache(cache_path: str | Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        ranges = {key: tuple(value) for key, value in metadata["ranges"].items()}
        result = {
            "features": cache["features"].astype(np.float32),
            "targets": cache["targets"].astype(np.float32),
            "file_indices": cache["file_indices"],
            "bearing_names": cache["bearing_names"],
            "ranges": ranges,
            "metadata": metadata,
        }
        if cache["splits"].size:
            result["splits"] = cache["splits"]
        if cache["conditions"].size:
            result["conditions"] = cache["conditions"]
        return result


def training_artifact_paths(checkpoint_path: str | Path) -> dict[str, Path]:
    checkpoint = Path(checkpoint_path)
    return {
        "checkpoint": checkpoint,
        "standardizer": checkpoint.with_suffix(".standardizer.npz"),
        "config": checkpoint.with_suffix(".config.json"),
    }


def save_training_artifacts(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    config: Mapping,
) -> dict[str, Path]:
    paths = training_artifact_paths(checkpoint_path)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), paths["checkpoint"])
    np.savez_compressed(
        paths["standardizer"],
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )
    paths["config"].write_text(
        json.dumps(_json_safe(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def load_training_artifacts(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location=None,
) -> dict:
    paths = training_artifact_paths(checkpoint_path)
    state_dict = torch.load(paths["checkpoint"], map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    with np.load(paths["standardizer"], allow_pickle=False) as standardizer:
        mean = standardizer["mean"].astype(np.float32)
        std = standardizer["std"].astype(np.float32)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    return {
        "mean": mean,
        "std": std,
        "config": config,
        "paths": paths,
    }


def make_sequence_index(
    ranges: Mapping[str, tuple[int, int] | list[int]],
    sequence_length: int,
    sequence_step: int = 1,
    bearings: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if sequence_step <= 0:
        raise ValueError("sequence_step must be greater than 0")

    selected_bearings = list(bearings) if bearings is not None else list(ranges.keys())
    windows = []
    window_bearings = []
    for bearing_name in selected_bearings:
        start, end = ranges[bearing_name]
        for window_start in range(start, end - sequence_length + 1, sequence_step):
            windows.append((window_start, window_start + sequence_length))
            window_bearings.append(bearing_name)

    return np.asarray(windows, dtype=np.int64), np.asarray(window_bearings)


def fit_feature_standardizer(features: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(features.shape[0], dtype=bool)
    for start, end in windows:
        mask[start:end] = True
    selected = features[mask]
    mean = selected.mean(axis=0).astype(np.float32)
    std = selected.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class SequenceFeatureDataset(TorchDataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        windows: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.windows = np.asarray(windows, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        self.std = np.asarray(std, dtype=np.float32) if std is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.windows[index]
        x = self.features[start:end]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = self.targets[end - 1]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def _multi_hot_fault_label(faults: Sequence[str]) -> np.ndarray:
    label = np.zeros(len(XJTU_FAULT_TYPES), dtype=np.float32)
    for fault in faults:
        label[XJTU_FAULT_TYPES.index(fault)] = 1.0
    return label


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

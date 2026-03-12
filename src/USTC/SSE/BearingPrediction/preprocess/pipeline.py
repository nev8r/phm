"""
Preprocess pipeline module

this file is for composing configurable preprocessing transforms

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from USTC.SSE.BearingPrediction.common.registry import ComponentRegistry

PREPROCESSOR_REGISTRY = ComponentRegistry("preprocessor")


class Transform:
    """
    Base class for signal transforms.
    """

    def __call__(self, signal_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@PREPROCESSOR_REGISTRY.register("zscore")
class ZScoreNormalize(Transform):
    """
    Z-score normalization transform.
    """

    def __call__(self, signal_values: np.ndarray) -> np.ndarray:
        mean_value = float(np.mean(signal_values))
        std_value = float(np.std(signal_values))
        if std_value == 0.0:
            return signal_values - mean_value
        return (signal_values - mean_value) / std_value


@PREPROCESSOR_REGISTRY.register("minmax")
class MinMaxNormalize(Transform):
    """
    Min-max normalization transform.
    """

    def __call__(self, signal_values: np.ndarray) -> np.ndarray:
        min_value = float(np.min(signal_values))
        max_value = float(np.max(signal_values))
        span = max_value - min_value
        if span == 0.0:
            return signal_values - min_value
        return (signal_values - min_value) / span


@PREPROCESSOR_REGISTRY.register("robust_clip")
class RobustClip(Transform):
    """
    Robust median absolute deviation clipping.
    """

    def __init__(self, clip_sigma: float = 3.0) -> None:
        self.clip_sigma = clip_sigma

    def __call__(self, signal_values: np.ndarray) -> np.ndarray:
        median_value = float(np.median(signal_values))
        mad_value = float(np.median(np.abs(signal_values - median_value)))
        if mad_value == 0.0:
            return signal_values
        scale = 1.4826 * mad_value
        lower_bound = median_value - (self.clip_sigma * scale)
        upper_bound = median_value + (self.clip_sigma * scale)
        return np.clip(signal_values, lower_bound, upper_bound)


@dataclass
class SlidingWindowConfig:
    """
    Sliding window parameters.

    Parameters
    ----------
    window_size : int
        window size
    stride : int
        stride
    """

    window_size: int
    stride: int


class SlidingWindowSegmenter:
    """
    Convert long signals into overlapping windows.
    """

    def __init__(self, config: SlidingWindowConfig) -> None:
        self.config = config

    def segment(self, signal_values: np.ndarray) -> list[np.ndarray]:
        """
        segment a long signal into windows

        Parameters
        ----------
        signal_values : np.ndarray
            signal values

        Returns
        -------
        list[np.ndarray]
            segmented windows
        """

        windows: list[np.ndarray] = []
        if signal_values.size < self.config.window_size:
            padded_signal = np.pad(
                signal_values,
                (0, self.config.window_size - signal_values.size),
                mode="edge",
            )
            return [padded_signal.astype(np.float32)]

        last_start = signal_values.size - self.config.window_size
        for start_index in range(0, last_start + 1, self.config.stride):
            end_index = start_index + self.config.window_size
            windows.append(signal_values[start_index:end_index].astype(np.float32))
        return windows


class PreprocessingPipeline:
    """
    Sequential preprocessing pipeline.
    """

    def __init__(self, transforms: Iterable[Transform]) -> None:
        self.transforms = list(transforms)

    def apply(self, signal_values: np.ndarray) -> np.ndarray:
        """
        apply each transform in order

        Parameters
        ----------
        signal_values : np.ndarray
            input signal

        Returns
        -------
        np.ndarray
            transformed signal
        """

        output_values = np.asarray(signal_values, dtype=np.float32)
        for transform in self.transforms:
            output_values = np.asarray(transform(output_values), dtype=np.float32)
        return output_values


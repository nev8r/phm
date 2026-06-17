"""
Feature backend module

this file is for selecting pluggable bearing feature extraction backends

created by cyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.feature.engineering import FeatureConfig, SignalFeatureExtractor


DEFAULT_FEATURE_SAMPLE_RATE = 25_600.0


class FeatureBackend(Protocol):
    """
    Common feature backend interface.
    """

    def extract(self, windows: list[np.ndarray]) -> pd.DataFrame:
        """
        extract feature records from signal windows

        Parameters
        ----------
        windows : list[np.ndarray]
            signal windows

        Returns
        -------
        pd.DataFrame
            feature records ordered like input windows
        """


@dataclass(init=False)
class FeatureBackendConfig:
    """
    Feature backend configuration.

    Parameters
    ----------
    backend : str
        backend name; interchangeable with name during initialization
    sample_rate : float
        signal sample rate
    """

    backend: str = "manual_19"
    sample_rate: float = DEFAULT_FEATURE_SAMPLE_RATE

    def __init__(
        self,
        backend: str | None = None,
        *,
        name: str | None = None,
        sample_rate: float = DEFAULT_FEATURE_SAMPLE_RATE,
    ) -> None:
        if backend is not None and name is not None and backend != name:
            raise ValueError("backend and name must refer to the same feature backend")
        self.backend = backend or name or "manual_19"
        self.sample_rate = sample_rate

    @property
    def name(self) -> str:
        """
        return backend name alias

        Returns
        -------
        str
            backend name
        """

        return self.backend


class ManualFeatureBackend:
    """
    Extract the default 19 handcrafted signal features.
    """

    def __init__(self, sample_rate: float) -> None:
        self.config = FeatureConfig(sample_rate=sample_rate)
        self.extractor = SignalFeatureExtractor(self.config)

    def extract(self, windows: list[np.ndarray]) -> pd.DataFrame:
        """
        extract manual features

        Parameters
        ----------
        windows : list[np.ndarray]
            signal windows

        Returns
        -------
        pd.DataFrame
            manual feature table
        """

        feature_frame = self.extractor.extract([np.asarray(window_values, dtype=float) for window_values in windows])
        feature_frame = feature_frame.reindex(columns=list(self.config.enabled_features))
        return _clean_feature_frame(feature_frame)


class TsfreshFeatureBackend:
    """
    Extract tsfresh features from signal windows.
    """

    def __init__(self, feature_set: str, sample_rate: float) -> None:
        self.feature_set = _normalize_tsfresh_feature_set(feature_set)
        self.sample_rate = sample_rate

    def extract(self, windows: list[np.ndarray]) -> pd.DataFrame:
        """
        extract tsfresh features

        Parameters
        ----------
        windows : list[np.ndarray]
            signal windows

        Returns
        -------
        pd.DataFrame
            tsfresh feature table ordered like input windows
        """

        if not windows:
            return pd.DataFrame()

        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError(
                "tsfresh feature backend requires the advanced extra. "
                "Run: uv run --extra advanced bearing-prediction"
            ) from exc

        feature_parameters = MinimalFCParameters() if self.feature_set == "minimal" else EfficientFCParameters()
        long_frame = _build_tsfresh_long_frame(windows)
        feature_frame = extract_features(
            long_frame,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=feature_parameters,
            disable_progressbar=True,
            n_jobs=1,
        )
        feature_frame.index = feature_frame.index.astype(int)
        feature_frame = feature_frame.sort_index().reindex(range(len(windows))).reset_index(drop=True)
        return _clean_feature_frame(feature_frame)


class CompositeFeatureBackend:
    """
    Concatenate manual and tsfresh features.
    """

    def __init__(self, tsfresh_feature_set: str, sample_rate: float) -> None:
        self.manual_backend = ManualFeatureBackend(sample_rate)
        self.tsfresh_backend = TsfreshFeatureBackend(tsfresh_feature_set, sample_rate)

    def extract(self, windows: list[np.ndarray]) -> pd.DataFrame:
        """
        extract concatenated manual and tsfresh features

        Parameters
        ----------
        windows : list[np.ndarray]
            signal windows

        Returns
        -------
        pd.DataFrame
            composite feature table
        """

        manual_frame = self.manual_backend.extract(windows).reset_index(drop=True)
        tsfresh_frame = self.tsfresh_backend.extract(windows).add_prefix("tsfresh__").reset_index(drop=True)
        return _clean_feature_frame(pd.concat([manual_frame, tsfresh_frame], axis=1))


FeatureBackendInput = str | FeatureBackendConfig | FeatureBackend | None


def create_feature_backend(
    config_or_name: FeatureBackendInput = None,
    *,
    sample_rate: float | None = None,
) -> FeatureBackend:
    """
    create a feature backend from configuration or backend name

    Parameters
    ----------
    config_or_name : str | FeatureBackendConfig | FeatureBackend | None
        backend name, configuration, existing backend instance, or None
    sample_rate : float | None
        signal sample rate override

    Returns
    -------
    FeatureBackend
        configured feature backend
    """

    if config_or_name is not None and hasattr(config_or_name, "extract"):
        return config_or_name

    if isinstance(config_or_name, FeatureBackendConfig):
        backend_name = config_or_name.backend
        resolved_sample_rate = sample_rate if sample_rate is not None else config_or_name.sample_rate
    else:
        backend_name = str(config_or_name or "manual_19")
        resolved_sample_rate = sample_rate if sample_rate is not None else DEFAULT_FEATURE_SAMPLE_RATE

    normalized_name = _normalize_backend_name(backend_name)
    if normalized_name == "manual_19":
        return ManualFeatureBackend(resolved_sample_rate)
    if normalized_name == "tsfresh_minimal":
        return TsfreshFeatureBackend("minimal", resolved_sample_rate)
    if normalized_name == "tsfresh_efficient":
        return TsfreshFeatureBackend("efficient", resolved_sample_rate)
    if normalized_name == "manual_19_plus_tsfresh_minimal":
        return CompositeFeatureBackend("minimal", resolved_sample_rate)
    if normalized_name == "manual_19_plus_tsfresh_efficient":
        return CompositeFeatureBackend("efficient", resolved_sample_rate)
    raise ValueError(f"unsupported feature backend: {backend_name}")


def _build_tsfresh_long_frame(windows: list[np.ndarray]) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for window_index, window_values in enumerate(windows):
        signal_values = np.asarray(window_values, dtype=float).reshape(-1)
        records.extend(
            {
                "id": window_index,
                "time": int(time_index),
                "value": float(signal_value),
            }
            for time_index, signal_value in enumerate(signal_values)
        )
    return pd.DataFrame.from_records(records)


def _clean_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    cleaned_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cleaned_frame.astype(float)


def _normalize_backend_name(backend_name: str) -> str:
    normalized = backend_name.strip().lower().replace("-", "_")
    aliases = {
        "manual": "manual_19",
        "manual_19": "manual_19",
        "tsfresh_minimal": "tsfresh_minimal",
        "tsfresh_efficient": "tsfresh_efficient",
        "manual_19_plus_tsfresh_minimal": "manual_19_plus_tsfresh_minimal",
        "manual_19_plus_tsfresh_efficient": "manual_19_plus_tsfresh_efficient",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported feature backend: {backend_name}")
    return aliases[normalized]


def _normalize_tsfresh_feature_set(feature_set: str) -> str:
    normalized = feature_set.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "minimal": "minimal",
        "minimalfcparameters": "minimal",
        "efficient": "efficient",
        "efficientfcparameters": "efficient",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported tsfresh feature set: {feature_set}")
    return aliases[normalized]

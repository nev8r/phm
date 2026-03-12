"""
Project config module

this file is for managing project level configuration

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    Project file system paths.

    Parameters
    ----------
    project_root : Path
        repository root path
    data_dir : Path
        data directory path
    output_dir : Path
        output directory path
    """

    project_root: Path
    data_dir: Path
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "ProjectPaths":
        """
        build standard project paths from repository root

        Parameters
        ----------
        project_root : Path
            repository root path

        Returns
        -------
        ProjectPaths
            configured project paths
        """

        return cls(
            project_root=project_root,
            data_dir=project_root / "data",
            output_dir=project_root / "outputs",
        )


@dataclass(frozen=True)
class DataConfig:
    """
    Data processing configuration.

    Parameters
    ----------
    bearing_count : int
        number of synthetic bearings
    signal_length : int
        vibration signal length for each cycle
    min_failure_cycle : int
        minimum failure cycle
    max_failure_cycle : int
        maximum failure cycle
    window_size : int
        sliding window size
    window_stride : int
        sliding window stride
    sample_rate : float
        synthetic sampling rate
    random_state : int
        random seed
    """

    bearing_count: int = 12
    signal_length: int = 256
    min_failure_cycle: int = 80
    max_failure_cycle: int = 150
    window_size: int = 64
    window_stride: int = 32
    sample_rate: float = 256.0
    random_state: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """
    Model training configuration.

    Parameters
    ----------
    test_size : float
        holdout ratio
    random_state : int
        random seed
    prediction_horizon : int
        horizon used for failure probability prediction
    """

    test_size: float = 0.25
    random_state: int = 42
    prediction_horizon: int = 20


@dataclass(frozen=True)
class VisualizationConfig:
    """
    Visualization configuration.

    Parameters
    ----------
    figure_dpi : int
        figure dpi
    style : str
        seaborn style name
    """

    figure_dpi: int = 150
    style: str = "whitegrid"


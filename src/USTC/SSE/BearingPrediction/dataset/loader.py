"""
BearingDatasetLoader class

this class is for loading bearing dataset

created by cyy

copyright USTC

2026
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.config import DataConfig


@dataclass(frozen=True)
class BearingMeasurement:
    """
    A single bearing measurement record.

    Parameters
    ----------
    bearing_id : str
        bearing identifier
    cycle : int
        measurement cycle
    failure_cycle : int
        true failure cycle
    temperature : float
        operating temperature
    load : float
        operating load
    health_index : float
        synthetic health indicator
    signal : np.ndarray
        vibration signal
    """

    bearing_id: str
    cycle: int
    failure_cycle: int
    temperature: float
    load: float
    health_index: float
    signal: np.ndarray


class BearingDatasetLoader:
    """
    Load bearing data from csv files or generate a synthetic dataset.
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config
        self.random_generator = np.random.default_rng(data_config.random_state)

    def load_dataset(self, raw_data_dir: Path, generated_dir: Path) -> pd.DataFrame:
        """
        load dataset from csv files or create a synthetic fallback dataset

        Parameters
        ----------
        raw_data_dir : Path
            directory that may contain custom csv files
        generated_dir : Path
            directory for generated dataset export

        Returns
        -------
        pd.DataFrame
            cycle level bearing dataset
        """

        csv_paths = sorted(raw_data_dir.glob("*.csv"))
        if csv_paths:
            return self._load_csv_dataset(csv_paths)

        dataset = self._generate_synthetic_dataset()
        generated_dir.mkdir(parents=True, exist_ok=True)
        self._export_generated_dataset(dataset, generated_dir / "synthetic_bearing_dataset.csv")
        return dataset

    def _load_csv_dataset(self, csv_paths: list[Path]) -> pd.DataFrame:
        """
        load user provided csv dataset

        Parameters
        ----------
        csv_paths : list[Path]
            list of csv files

        Returns
        -------
        pd.DataFrame
            parsed dataset
        """

        dataset_frames: list[pd.DataFrame] = []
        required_columns = {"bearing_id", "cycle", "failure_cycle", "temperature", "load", "signal"}

        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            missing_columns = required_columns.difference(frame.columns)
            if missing_columns:
                missing_text = ", ".join(sorted(missing_columns))
                raise ValueError(f"missing required columns in {csv_path.name}: {missing_text}")

            parsed_frame = frame.copy()
            parsed_frame["signal"] = parsed_frame["signal"].apply(self._parse_signal)
            parsed_frame["health_index"] = parsed_frame.get(
                "health_index",
                1.0 - (parsed_frame["cycle"] / parsed_frame["failure_cycle"]).clip(upper=1.0),
            )
            parsed_frame["rul"] = (parsed_frame["failure_cycle"] - parsed_frame["cycle"]).clip(lower=0)
            parsed_frame["duration"] = parsed_frame["rul"] + 1
            if "event" in parsed_frame.columns:
                parsed_frame["event"] = parsed_frame["event"].astype(int)
            else:
                parsed_frame["event"] = 1
            dataset_frames.append(parsed_frame)

        dataset = pd.concat(dataset_frames, ignore_index=True)
        return dataset.sort_values(["bearing_id", "cycle"]).reset_index(drop=True)

    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        """
        generate a synthetic bearing run-to-failure dataset

        Returns
        -------
        pd.DataFrame
            generated dataset
        """

        records: list[dict[str, object]] = []
        time_axis = np.linspace(0.0, 1.0, self.data_config.signal_length, endpoint=False)

        for bearing_number in range(1, self.data_config.bearing_count + 1):
            failure_cycle = int(
                self.random_generator.integers(
                    self.data_config.min_failure_cycle,
                    self.data_config.max_failure_cycle + 1,
                )
            )
            base_frequency = float(self.random_generator.uniform(8.0, 18.0))
            load_level = float(self.random_generator.uniform(0.82, 1.18))

            for cycle in range(1, failure_cycle + 1):
                degradation = cycle / failure_cycle
                bearing_signal = self._build_signal(
                    time_axis=time_axis,
                    degradation=degradation,
                    base_frequency=base_frequency,
                    load_level=load_level,
                )
                temperature = 33.0 + (24.0 * degradation) + self.random_generator.normal(0.0, 0.8)
                health_index = float(np.clip(1.02 - (degradation ** 1.4), 0.0, 1.0))
                records.append(
                    {
                        "bearing_id": f"B{bearing_number:02d}",
                        "cycle": cycle,
                        "failure_cycle": failure_cycle,
                        "temperature": round(float(temperature), 4),
                        "load": round(load_level, 4),
                        "health_index": round(health_index, 4),
                        "signal": bearing_signal,
                        "rul": failure_cycle - cycle,
                        "duration": (failure_cycle - cycle) + 1,
                        "event": 1,
                    }
                )

        dataset = pd.DataFrame.from_records(records)
        return dataset.sort_values(["bearing_id", "cycle"]).reset_index(drop=True)

    def _build_signal(
        self,
        time_axis: np.ndarray,
        degradation: float,
        base_frequency: float,
        load_level: float,
    ) -> np.ndarray:
        """
        build a synthetic degradation dependent vibration signal

        Parameters
        ----------
        time_axis : np.ndarray
            time axis
        degradation : float
            degradation ratio
        base_frequency : float
            signal base frequency
        load_level : float
            load coefficient

        Returns
        -------
        np.ndarray
            generated vibration signal
        """

        amplitude = 0.8 + (2.6 * (degradation ** 1.8))
        fault_frequency = base_frequency + (12.0 * degradation)
        resonance_frequency = 3.0 * base_frequency
        white_noise = self.random_generator.normal(0.0, 0.12 + (0.35 * degradation), size=time_axis.size)
        impulse_mask = self.random_generator.random(time_axis.size) < (0.015 + (0.06 * degradation))
        impulses = impulse_mask.astype(float) * self.random_generator.normal(2.0 * degradation, 0.4, size=time_axis.size)

        signal = (
            amplitude * np.sin(2.0 * np.pi * fault_frequency * time_axis)
            + (0.35 * load_level) * np.sin(2.0 * np.pi * resonance_frequency * time_axis)
            + (0.18 * np.cos(2.0 * np.pi * (fault_frequency / 2.0) * time_axis))
            + white_noise
            + impulses
        )
        return signal.astype(float)

    def _export_generated_dataset(self, dataset: pd.DataFrame, output_path: Path) -> None:
        """
        export generated dataset for inspection

        Parameters
        ----------
        dataset : pd.DataFrame
            dataset to export
        output_path : Path
            output csv path
        """

        export_frame = dataset.copy()
        export_frame["signal"] = export_frame["signal"].apply(lambda values: json.dumps([round(float(item), 6) for item in values]))
        export_frame.to_csv(output_path, index=False)

    @staticmethod
    def _parse_signal(signal_text: str) -> np.ndarray:
        """
        parse signal string from csv

        Parameters
        ----------
        signal_text : str
            serialized signal

        Returns
        -------
        np.ndarray
            parsed signal values
        """

        stripped_text = signal_text.strip()
        if stripped_text.startswith("["):
            parsed_values = json.loads(stripped_text)
            return np.asarray(parsed_values, dtype=float)

        return np.asarray([float(value) for value in stripped_text.split()], dtype=float)

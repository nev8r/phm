"""
Open-source SOTA protocol module

this file is for defining SOTA target and reproduction evidence schemas

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


TARGET_COLUMNS = [
    "target_id",
    "method_name",
    "dataset_name",
    "condition_name",
    "metric_name",
    "target_value",
    "metric_direction",
    "source_type",
    "source_url",
    "source_commit",
    "split_description",
    "reproducibility_status",
    "license_name",
    "run_command",
    "notes",
]


REPRODUCTION_COLUMNS = [
    "target_id",
    "experiment_name",
    "method_name",
    "local_method_name",
    "dataset_name",
    "condition_name",
    "metric_name",
    "target_value",
    "local_value",
    "local_mean",
    "local_std",
    "gap_percent",
    "mean_gap_percent",
    "metric_direction",
    "run_count",
    "seeds",
    "prediction_count",
    "source_url",
    "source_commit",
    "split_description",
    "evidence_path",
    "status",
    "notes",
]


@dataclass(frozen=True)
class SotaTargetRecord:
    """
    Open-source SOTA target record.

    Parameters
    ----------
    target_id : str
        stable target identifier
    method_name : str
        reference method name
    dataset_name : str
        dataset name
    condition_name : str
        condition or split name
    metric_name : str
        metric used for comparison
    target_value : float
        reference metric value
    metric_direction : str
        lower or higher
    source_type : str
        source category
    source_url : str
        source repository or paper url
    source_commit : str
        repository commit or explicit reference marker
    split_description : str
        train/test split details
    reproducibility_status : str
        reproducibility assessment
    license_name : str
        license or paper marker
    run_command : str
        reproduction command when available
    notes : str
        extra caveats
    """

    target_id: str
    method_name: str
    dataset_name: str
    condition_name: str
    metric_name: str
    target_value: float
    metric_direction: str
    source_type: str
    source_url: str
    source_commit: str
    split_description: str
    reproducibility_status: str
    license_name: str
    run_command: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """
        convert record to plain dict

        Returns
        -------
        dict[str, object]
            row dictionary
        """

        return asdict(self)


@dataclass(frozen=True)
class SotaReproductionRecord:
    """
    Local SOTA reproduction or gap record.
    """

    target_id: str
    experiment_name: str
    method_name: str
    local_method_name: str
    dataset_name: str
    condition_name: str
    metric_name: str
    target_value: float
    local_value: float
    local_mean: float
    local_std: float
    gap_percent: float
    mean_gap_percent: float
    metric_direction: str
    run_count: int
    seeds: str
    prediction_count: int
    source_url: str
    source_commit: str
    split_description: str
    evidence_path: str
    status: str
    notes: str

    @classmethod
    def from_target(
        cls,
        target: SotaTargetRecord,
        *,
        experiment_name: str,
        local_method_name: str,
        local_value: float,
        local_mean: float,
        local_std: float,
        run_count: int,
        seeds: str,
        prediction_count: int,
        evidence_path: str,
        status: str,
        notes: str,
    ) -> "SotaReproductionRecord":
        """
        build reproduction record from target.

        Parameters
        ----------
        target : SotaTargetRecord
            reference target

        Returns
        -------
        SotaReproductionRecord
            reproduction row
        """

        higher_is_better = target.metric_direction == "higher"
        return cls(
            target_id=target.target_id,
            experiment_name=experiment_name,
            method_name=target.method_name,
            local_method_name=local_method_name,
            dataset_name=target.dataset_name,
            condition_name=target.condition_name,
            metric_name=target.metric_name,
            target_value=float(target.target_value),
            local_value=float(local_value),
            local_mean=float(local_mean),
            local_std=float(local_std),
            gap_percent=calculate_gap_percent(
                local_value=local_value,
                target_value=target.target_value,
                higher_is_better=higher_is_better,
            ),
            mean_gap_percent=calculate_gap_percent(
                local_value=local_mean,
                target_value=target.target_value,
                higher_is_better=higher_is_better,
            ),
            metric_direction=target.metric_direction,
            run_count=int(run_count),
            seeds=seeds,
            prediction_count=int(prediction_count),
            source_url=target.source_url,
            source_commit=target.source_commit,
            split_description=target.split_description,
            evidence_path=evidence_path,
            status=status,
            notes=notes,
        )

    def to_dict(self) -> dict[str, object]:
        """
        convert record to plain dict

        Returns
        -------
        dict[str, object]
            row dictionary
        """

        return asdict(self)


def calculate_gap_percent(*, local_value: float, target_value: float, higher_is_better: bool) -> float:
    """
    calculate relative gap to a SOTA target.

    Parameters
    ----------
    local_value : float
        local metric value
    target_value : float
        target metric value
    higher_is_better : bool
        metric direction flag

    Returns
    -------
    float
        non-negative gap percentage
    """

    if pd.isna(local_value) or pd.isna(target_value):
        return float("nan")
    if abs(target_value) < 1e-12:
        raise ValueError("target_value must be non-zero for gap calculation")
    if higher_is_better:
        return max(0.0, float((target_value - local_value) / abs(target_value) * 100.0))
    return max(0.0, float((local_value - target_value) / abs(target_value) * 100.0))


def validate_target_frame(target_frame: pd.DataFrame) -> None:
    """
    validate SOTA target table.

    Parameters
    ----------
    target_frame : pd.DataFrame
        target table
    """

    _require_columns(target_frame, TARGET_COLUMNS, frame_name="target_frame")
    for row_index, row in target_frame.iterrows():
        for column_name in [
            "target_id",
            "method_name",
            "dataset_name",
            "metric_name",
            "metric_direction",
            "source_url",
            "source_commit",
            "split_description",
            "reproducibility_status",
        ]:
            if pd.isna(row[column_name]) or str(row[column_name]).strip() == "":
                raise ValueError(f"{column_name} is required for SOTA target row {row_index}")
        if row["metric_direction"] not in {"lower", "higher"}:
            raise ValueError(f"metric_direction must be lower or higher for row {row_index}")
        if float(row["target_value"]) <= 0.0:
            raise ValueError(f"target_value must be positive for row {row_index}")


def validate_reproduction_frame(reproduction_frame: pd.DataFrame, *, min_run_count: int = 1) -> None:
    """
    validate SOTA reproduction table.

    Parameters
    ----------
    reproduction_frame : pd.DataFrame
        reproduction table
    min_run_count : int
        minimum repeated run count
    """

    _require_columns(reproduction_frame, REPRODUCTION_COLUMNS, frame_name="reproduction_frame")
    for row_index, row in reproduction_frame.iterrows():
        for column_name in ["target_id", "experiment_name", "local_method_name", "source_url", "evidence_path", "status"]:
            if pd.isna(row[column_name]) or str(row[column_name]).strip() == "":
                raise ValueError(f"{column_name} is required for reproduction row {row_index}")
        if int(row["prediction_count"]) < 0:
            raise ValueError(f"prediction_count must be non-negative for reproduction row {row_index}")
        status = str(row["status"])
        if status.startswith("BLOCKED") or "EXTERNAL_ENV" in status or status == "REFERENCE_ONLY":
            if int(row["run_count"]) != 0:
                raise ValueError(f"run_count must be 0 for non-run reproduction row {row_index}")
            if "EXTERNAL_ENV" in status and str(row["evidence_path"]) == "not_available":
                raise ValueError(f"external environment row {row_index} must include attempt evidence_path")
            continue
        if int(row["run_count"]) < min_run_count:
            raise ValueError(f"run_count must be at least {min_run_count} for reproduction row {row_index}")
        for column_name in ["target_value", "local_value", "local_mean", "gap_percent", "mean_gap_percent"]:
            if pd.isna(row[column_name]):
                raise ValueError(f"{column_name} must be numeric for reproduction row {row_index}")


def _require_columns(frame: pd.DataFrame, required_columns: list[str], *, frame_name: str) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in frame.columns]
    if missing_columns:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing_columns)}")

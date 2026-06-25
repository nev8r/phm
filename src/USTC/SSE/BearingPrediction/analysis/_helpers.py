"""
Shared helpers for feature analysis.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS


def feature_columns(features: pd.DataFrame) -> List[str]:
    return [column for column in features.columns if column not in FEATURE_INDEX_COLUMNS and not column.startswith("__")]


def aligned_frame(features: pd.DataFrame, labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if labels is None:
        return features.copy()
    return features.merge(labels, on="sample_uid", suffixes=("", "__label"), how="inner")


def split_names(sample_uids: pd.Series, split_result) -> pd.Series:
    if split_result is None:
        return pd.Series(["all"] * len(sample_uids), index=sample_uids.index)
    mapping: Dict[str, str] = {}
    mapping.update({sample_uid: "train" for sample_uid in split_result.train_sample_uids})
    mapping.update({sample_uid: "val" for sample_uid in split_result.val_sample_uids})
    mapping.update({sample_uid: "test" for sample_uid in split_result.test_sample_uids})
    return sample_uids.map(mapping).fillna("unused")


def fit_subset(data: pd.DataFrame, split_result, fit_scope: str) -> pd.DataFrame:
    if split_result is None:
        return data.copy()
    data = data.copy()
    data["__split"] = split_names(data["sample_uid"], split_result)
    if fit_scope == "train_only":
        return data[data["__split"] == "train"].drop(columns=["__split"]).copy()
    if fit_scope in {"all", "all_no_split"}:
        return data[data["__split"] != "unused"].drop(columns=["__split"]).copy()
    raise ValueError(f"Unsupported analysis fit_scope: {fit_scope}")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.DataFrame({"x": safe_numeric(x), "y": safe_numeric(y)}).dropna()
    if len(data) < 2 or data["x"].nunique() <= 1 or data["y"].nunique() <= 1:
        return 0.0
    value = data["x"].corr(data["y"], method=method)
    if pd.isna(value):
        return 0.0
    return float(value)


def minmax(values: pd.Series) -> pd.Series:
    values = safe_numeric(values).fillna(0.0)
    low = float(values.min())
    high = float(values.max())
    denominator = high - low
    if denominator <= np.finfo(float).eps:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / denominator


def label_source_features(hi: Optional[pd.DataFrame], fpt: Optional[Dict]) -> Set[str]:
    del fpt
    if hi is None or "hi_source_column" not in hi.columns or hi.empty:
        return set()
    return {str(value) for value in hi["hi_source_column"].dropna().unique()}


def present_columns(frame: Optional[pd.DataFrame], columns: Iterable[str]) -> bool:
    return frame is not None and all(column in frame.columns for column in columns)

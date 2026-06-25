"""
Feature distribution summary analyzer.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.analysis._helpers import feature_columns, safe_numeric, split_names


class FeatureSummaryAnalyzer:
    def analyze(self, features: pd.DataFrame, split_result=None) -> pd.DataFrame:
        data = features.copy()
        data["__split"] = split_names(data["sample_uid"], split_result)
        rows: List[Dict] = []
        splits = [split for split in ["train", "val", "test", "all"] if split == "all" or (data["__split"] == split).any()]
        for split in splits:
            split_data = data if split == "all" else data[data["__split"] == split]
            for column in feature_columns(features):
                values = safe_numeric(split_data[column])
                finite = values[np.isfinite(values)]
                rows.append({
                    "feature": column,
                    "split": split,
                    "count": int(len(values)),
                    "missing_count": int(values.isna().sum()),
                    "nan_count": int(values.isna().sum()),
                    "inf_count": int(np.isinf(pd.to_numeric(split_data[column], errors="coerce")).sum()),
                    "mean": float(finite.mean()) if len(finite) else 0.0,
                    "std": float(finite.std(ddof=0)) if len(finite) else 0.0,
                    "min": float(finite.min()) if len(finite) else 0.0,
                    "p25": float(finite.quantile(0.25)) if len(finite) else 0.0,
                    "median": float(finite.median()) if len(finite) else 0.0,
                    "p75": float(finite.quantile(0.75)) if len(finite) else 0.0,
                    "max": float(finite.max()) if len(finite) else 0.0,
                    "variance": float(finite.var(ddof=0)) if len(finite) else 0.0,
                    "is_constant": bool(finite.nunique(dropna=True) <= 1),
                })
        return pd.DataFrame(rows)

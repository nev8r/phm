"""
RUL feature correlation analyzer.

Purpose: analyze experiment outputs and generate reviewable reports
Author: zyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.analysis._helpers import aligned_frame, feature_columns, safe_corr


class RulFeatureAnalyzer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def analyze(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        target_column = str(OmegaConf.select(self.cfg, "target_column", default="piecewise_rul_norm"))
        if target_column not in labels.columns:
            return _empty()
        data = aligned_frame(features, labels[["sample_uid", target_column]])
        methods = list(OmegaConf.select(self.cfg, "methods", default=["pearson", "spearman", "kendall"]))
        rows: List[Dict] = []
        for column in feature_columns(features):
            row = {
                "feature": column,
                "target_column": target_column,
                "split_scope": "fit",
                "n": int(data[[column, target_column]].dropna().shape[0]),
            }
            for method in ["pearson", "spearman", "kendall"]:
                value = safe_corr(data[column], data[target_column], method) if method in methods else 0.0
                row[method] = value
                row[f"abs_{method}"] = abs(value)
            rows.append(row)
        return pd.DataFrame(rows)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "feature", "target_column", "split_scope", "pearson", "spearman", "kendall",
        "abs_pearson", "abs_spearman", "abs_kendall", "n",
    ])

"""
Early-fault feature sensitivity analyzer.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from USTC.SSE.BearingPrediction.analysis._helpers import aligned_frame, feature_columns, safe_numeric


class EarlyFaultFeatureAnalyzer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def analyze(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        target_column = str(OmegaConf.select(self.cfg, "target_column", default="early_fault"))
        if target_column not in labels.columns:
            return _empty()
        data = aligned_frame(features, labels[["sample_uid", target_column]]).dropna(subset=[target_column])
        y = data[target_column].astype(int)
        rows: List[Dict] = []
        for column in feature_columns(features):
            x = safe_numeric(data[column]).fillna(0.0)
            healthy = x[y == 0]
            fault = x[y == 1]
            auc = _direction_free_auc(y, x)
            rows.append({
                "feature": column,
                "healthy_mean": float(healthy.mean()) if len(healthy) else 0.0,
                "fault_mean": float(fault.mean()) if len(fault) else 0.0,
                "mean_shift": float(fault.mean() - healthy.mean()) if len(healthy) and len(fault) else 0.0,
                "cohens_d": _cohens_d(healthy, fault),
                "auc": auc,
            })
        return pd.DataFrame(rows)


def _cohens_d(healthy: pd.Series, fault: pd.Series) -> float:
    if len(healthy) == 0 or len(fault) == 0:
        return 0.0
    pooled = np.sqrt((healthy.var(ddof=0) + fault.var(ddof=0)) / 2.0)
    if pooled <= np.finfo(float).eps:
        return 0.0
    return float((fault.mean() - healthy.mean()) / pooled)


def _direction_free_auc(y: pd.Series, x: pd.Series) -> float:
    if y.nunique() < 2:
        return 0.5
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "healthy_mean", "fault_mean", "mean_shift", "cohens_d", "auc"])

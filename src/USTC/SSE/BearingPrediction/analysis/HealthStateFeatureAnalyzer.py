"""
Health-state feature separability analyzer.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from USTC.SSE.BearingPrediction.analysis._helpers import aligned_frame, feature_columns, safe_numeric


class HealthStateFeatureAnalyzer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def analyze(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        target_column = str(OmegaConf.select(self.cfg, "target_column", default="health_state_id"))
        if target_column not in labels.columns:
            return _empty()
        data = aligned_frame(features, labels[["sample_uid", target_column]]).dropna(subset=[target_column])
        y = data[target_column].astype(int)
        rows: List[Dict] = []
        for column in feature_columns(features):
            x = safe_numeric(data[column]).fillna(0.0)
            anova_f, anova_p = _anova(x, y)
            rows.append({
                "feature": column,
                "mutual_information": _mutual_information(x, y),
                "fisher_score": _fisher_score(x, y),
                "anova_f": anova_f,
                "anova_p": anova_p,
                "class_count": int(y.nunique()),
            })
        return pd.DataFrame(rows)


def _mutual_information(x: pd.Series, y: pd.Series) -> float:
    if y.nunique() <= 1 or len(y) < 3 or y.value_counts().min() < 2:
        return 0.0
    try:
        return float(mutual_info_classif(x.to_frame(), y, discrete_features=False, random_state=0)[0])
    except ValueError:
        return 0.0


def _fisher_score(x: pd.Series, y: pd.Series) -> float:
    overall_mean = float(x.mean())
    between = 0.0
    within = 0.0
    for label, group in x.groupby(y):
        del label
        between += len(group) * (float(group.mean()) - overall_mean) ** 2
        within += float(((group - float(group.mean())) ** 2).sum())
    return float(between / (within + 1.0e-12))


def _anova(x: pd.Series, y: pd.Series):
    groups = [group.to_numpy(dtype=float) for _, group in x.groupby(y) if len(group) > 0]
    if len(groups) < 2 or all(len(group) < 2 for group in groups):
        return 0.0, 1.0
    result = stats.f_oneway(*groups)
    return float(0.0 if np.isnan(result.statistic) else result.statistic), float(1.0 if np.isnan(result.pvalue) else result.pvalue)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "mutual_information", "fisher_score", "anova_f", "anova_p", "class_count"])

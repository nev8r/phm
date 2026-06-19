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
            row = {
                "feature": column,
                "mutual_information": _mutual_information(x, y),
                "fisher_score": _fisher_score(x, y),
                "anova_f": anova_f,
                "anova_p": anova_p,
                "class_count": int(y.nunique()),
            }
            row.update(_class_stats(x, y))
            rows.append(row)
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
    if all(float(np.std(group)) <= np.finfo(float).eps for group in groups):
        means = np.array([float(np.mean(group)) for group in groups])
        if len(np.unique(means)) > 1:
            return 1.0e12, 0.0
        return 0.0, 1.0
    result = stats.f_oneway(*groups)
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    if np.isnan(statistic):
        statistic = 0.0
    if np.isinf(statistic):
        statistic = 1.0e12
    if np.isnan(pvalue):
        pvalue = 1.0
    return statistic, pvalue


def _class_stats(x: pd.Series, y: pd.Series) -> Dict:
    stats_by_class: Dict[str, float] = {}
    for label, group in x.groupby(y):
        label_text = str(int(label)) if float(label).is_integer() else str(label)
        stats_by_class[f"state_{label_text}_mean"] = float(group.mean()) if len(group) else 0.0
        stats_by_class[f"state_{label_text}_std"] = float(group.std(ddof=0)) if len(group) else 0.0
    return stats_by_class


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["feature", "mutual_information", "fisher_score", "anova_f", "anova_p", "class_count"])

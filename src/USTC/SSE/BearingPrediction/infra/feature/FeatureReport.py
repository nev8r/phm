"""
Feature report builder.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.infra.feature.FeatureFrame import FEATURE_INDEX_COLUMNS


def build_feature_report(
        raw_features: pd.DataFrame,
        cleaned_features: pd.DataFrame,
        feature_set: str,
        backend_reports: List[Dict],
        cleaner,
        cleaner_fit_scope: str,
) -> Dict:
    raw_feature_columns = [column for column in raw_features.columns if column not in FEATURE_INDEX_COLUMNS]
    cleaned_feature_columns = [column for column in cleaned_features.columns if column not in FEATURE_INDEX_COLUMNS]
    raw_values = raw_features[raw_feature_columns].replace([np.inf, -np.inf], np.nan)
    num_nan = int(raw_values.isna().sum().sum())
    num_inf = int(np.isinf(raw_features[raw_feature_columns].to_numpy(dtype=float)).sum())

    return {
        "ok": True,
        "feature_set": feature_set,
        "num_samples": int(len(raw_features)),
        "num_raw_features": len(raw_feature_columns),
        "num_cleaned_features": len(cleaned_feature_columns),
        "num_dropped_features": len(cleaner.dropped_columns),
        "dropped_features": cleaner.dropped_columns,
        "num_nan_before_cleaning": num_nan,
        "num_inf_before_cleaning": num_inf,
        "cleaner_enabled": cleaner.enabled,
        "cleaner_fit_scope": cleaner_fit_scope,
        "backends": backend_reports,
    }

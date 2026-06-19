"""
Feature-analysis leakage checks.
"""

from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.analysis._helpers import label_source_features


class LeakageGuard:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def check(self, features: pd.DataFrame, hi: Optional[pd.DataFrame] = None, fpt: Optional[Dict] = None) -> Dict:
        del features
        fit_scope = str(OmegaConf.select(self.cfg, "fit_scope", default="train_only"))
        warnings: List[Dict] = []
        if fit_scope != "train_only":
            warnings.append({
                "type": "fit_scope",
                "message": f"Analysis fit_scope is {fit_scope}; feature ranking may use non-train data.",
            })
        for feature in sorted(label_source_features(hi, fpt)):
            warnings.append({
                "type": "label_source_feature",
                "feature": feature,
                "message": "Feature was used as HI source for FPT-based labels.",
            })
        return {"ok": True, "fit_scope": fit_scope, "warnings": warnings}

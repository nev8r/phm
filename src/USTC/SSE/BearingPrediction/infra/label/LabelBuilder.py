"""
Label builder orchestration.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: cyj
Program date: 2026-06
Copyright: USTC

2026
"""

from typing import Dict, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.degradation.FeatureColumnHICalculator import FeatureColumnHICalculator
from USTC.SSE.BearingPrediction.infra.degradation.ThreeSigmaFPTDetector import ThreeSigmaFPTDetector
from USTC.SSE.BearingPrediction.infra.label.LabelFrame import LABEL_INDEX_COLUMNS, LabelFrame
from USTC.SSE.BearingPrediction.infra.label.LabelRegistry import build_labeler
from USTC.SSE.BearingPrediction.infra.label.LabelReport import build_label_report
from USTC.SSE.BearingPrediction.infra.label.LabelSpec import LabelSpec


class LabelBuilder:
    """Construct task labels and optional degradation evidence from indexed samples."""

    def __init__(self, cfg: DictConfig):
        """Store the label config used for output labels and HI/FPT settings."""
        self.cfg = cfg

    def build(
            self,
            index: pd.DataFrame,
            raw_features: Optional[pd.DataFrame] = None,
            cleaned_features: Optional[pd.DataFrame] = None,
            split=None,
    ):
        """Build configured label columns and return labels, spec, report, HI, and FPT."""
        del split
        requires_features = bool(OmegaConf.select(self.cfg, "requires_features", default=False))
        features = raw_features if raw_features is not None else cleaned_features
        if requires_features and features is None:
            raise ValueError("label.requires_features=true requires feature data")

        hi_df = None
        fpt_payload = None
        fpt_map: Dict[Tuple[str, str], Dict] = {}
        if requires_features and OmegaConf.select(self.cfg, "hi", default=None) is not None:
            hi_df, fpt_payload = self._build_hi_fpt(features)
            fpt_map = _fpt_map(fpt_payload)

        labels = index[list(LABEL_INDEX_COLUMNS)].copy()
        fault_type_stage_mapping = None
        outputs = list(OmegaConf.select(self.cfg, "outputs", default=[]))
        for output_cfg in outputs:
            label_type = str(OmegaConf.select(output_cfg, "type", default=""))
            labeler = build_labeler(output_cfg)
            if label_type in {"piecewise_rul", "health_state", "early_fault"}:
                output = labeler.label(index, fpt_map)
            elif label_type == "fault_type_stage":
                if "health_state_id" not in labels.columns:
                    raise ValueError("fault_type_stage requires health_state labels")
                output, fault_type_stage_mapping = labeler.label(index, labels)
                if not [column for column in output.columns if column != "sample_uid"]:
                    continue
            else:
                output = labeler.label(index)
            labels = labels.merge(output, on="sample_uid", how="left")

        spec = LabelSpec(
            name=str(OmegaConf.select(self.cfg, "name", default="labels")),
            version=str(OmegaConf.select(self.cfg, "version", default="v1")),
            requires_features=requires_features,
            outputs=_to_plain(OmegaConf.select(self.cfg, "outputs", default=[])),
            hi=_to_plain(OmegaConf.select(self.cfg, "hi", default=None)),
            fpt=_to_plain(OmegaConf.select(self.cfg, "fpt", default=None)),
        ).to_dict()
        label_columns = [column for column in labels.columns if column not in LABEL_INDEX_COLUMNS]
        frame = LabelFrame(labels, list(LABEL_INDEX_COLUMNS), label_columns, spec)
        frame.validate()
        report = build_label_report(
            labels=labels,
            label_set=str(OmegaConf.select(self.cfg, "name", default="labels")),
            requires_features=requires_features,
            fpt_payload=fpt_payload,
            fault_type_stage_mapping=fault_type_stage_mapping,
        )
        return labels, spec, report, hi_df, fpt_payload

    def _build_hi_fpt(self, features: pd.DataFrame):
        """Calculate a health indicator and first predicting time payload for labelers."""
        hi_cfg = OmegaConf.select(self.cfg, "hi", default={})
        hi_method = str(OmegaConf.select(hi_cfg, "method", default="feature_column"))
        if hi_method != "feature_column":
            raise NotImplementedError(f"Unsupported HI method: {hi_method}")
        hi_frame = FeatureColumnHICalculator(hi_cfg).calculate(features)
        hi_df = hi_frame.data
        source_column = str(hi_df["hi_source_column"].iloc[0])

        fpt_cfg = OmegaConf.select(self.cfg, "fpt", default={})
        fpt_method = str(OmegaConf.select(fpt_cfg, "method", default="three_sigma"))
        if fpt_method != "three_sigma":
            raise NotImplementedError(f"Unsupported FPT method: {fpt_method}")
        fpt_payload = ThreeSigmaFPTDetector(fpt_cfg).detect(hi_df, source_column=source_column)
        return hi_df, fpt_payload


def _fpt_map(fpt_payload: Optional[Dict]) -> Dict[Tuple[str, str], Dict]:
    if not fpt_payload:
        return {}
    return {
        (str(result["dataset"]), str(result["bearing_id"])): result
        for result in fpt_payload.get("results", [])
    }


def _to_plain(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value

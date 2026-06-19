"""
Feature analysis orchestration.
"""

from typing import Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.analysis.AnalysisSpec import AnalysisSpec
from USTC.SSE.BearingPrediction.analysis.DegradationScoreAnalyzer import DegradationScoreAnalyzer
from USTC.SSE.BearingPrediction.analysis.EarlyFaultFeatureAnalyzer import EarlyFaultFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis.FaultTypeFeatureAnalyzer import FaultTypeFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis.FeatureRanking import build_feature_ranking
from USTC.SSE.BearingPrediction.analysis.FeatureSummaryAnalyzer import FeatureSummaryAnalyzer
from USTC.SSE.BearingPrediction.analysis.HealthStateFeatureAnalyzer import HealthStateFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis.LeakageGuard import LeakageGuard
from USTC.SSE.BearingPrediction.analysis.RulFeatureAnalyzer import RulFeatureAnalyzer
from USTC.SSE.BearingPrediction.analysis._helpers import feature_columns, fit_subset, label_source_features


class AnalysisBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def build(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            index: pd.DataFrame,
            split_result=None,
            hi: pd.DataFrame = None,
            fpt: dict = None,
    ) -> Dict:
        del index
        fit_scope = str(OmegaConf.select(self.cfg, "scope.fit_scope", default="train_only"))
        actual_fit_scope = "all_no_split" if split_result is None and fit_scope == "train_only" else fit_scope
        fit_features = fit_subset(features, split_result, actual_fit_scope)
        fit_labels = labels[labels["sample_uid"].isin(set(fit_features["sample_uid"]))].copy()
        sections = []

        feature_summary = None
        if bool(OmegaConf.select(self.cfg, "summary.enabled", default=False)):
            feature_summary = FeatureSummaryAnalyzer().analyze(features, split_result=split_result)
            sections.append("summary")

        rul_correlation = None
        if bool(OmegaConf.select(self.cfg, "rul_correlation.enabled", default=False)):
            rul_correlation = RulFeatureAnalyzer(OmegaConf.select(self.cfg, "rul_correlation", default={})).analyze(fit_features, fit_labels)
            sections.append("rul_correlation")

        degradation_scores = None
        if bool(OmegaConf.select(self.cfg, "degradation_scores.enabled", default=False)):
            degradation_scores = DegradationScoreAnalyzer(OmegaConf.select(self.cfg, "degradation_scores", default={})).analyze(fit_features)
            sections.append("degradation_scores")

        health_state_separability = None
        if bool(OmegaConf.select(self.cfg, "health_state.enabled", default=False)):
            health_state_separability = HealthStateFeatureAnalyzer(OmegaConf.select(self.cfg, "health_state", default={})).analyze(fit_features, fit_labels)
            sections.append("health_state")

        early_fault_scores = None
        if bool(OmegaConf.select(self.cfg, "early_fault.enabled", default=False)):
            early_fault_scores = EarlyFaultFeatureAnalyzer(OmegaConf.select(self.cfg, "early_fault", default={})).analyze(fit_features, fit_labels)
            sections.append("early_fault")

        fault_type_scores = None
        fault_type_skipped = False
        if bool(OmegaConf.select(self.cfg, "fault_type.enabled", default=False)):
            fault_type_cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.select(self.cfg, "fault_type"), resolve=True))
            target_column = str(OmegaConf.select(self.cfg, "fault_type.target_column", default="fault_type_stage_id"))
            if target_column in fit_labels.columns:
                fault_type_scores = FaultTypeFeatureAnalyzer(fault_type_cfg).analyze(fit_features, fit_labels)
                sections.append("fault_type")
            else:
                fault_type_skipped = True

        leakage_cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.select(self.cfg, "leakage", default={}), resolve=True) or {})
        leakage_cfg["fit_scope"] = actual_fit_scope
        leakage_report = LeakageGuard(leakage_cfg).check(fit_features, hi=hi, fpt=fpt)
        source_features = label_source_features(hi, fpt)
        feature_ranking = build_feature_ranking(
            feature_columns=feature_columns(features),
            rul_correlation=rul_correlation,
            degradation_scores=degradation_scores,
            health_state_separability=health_state_separability,
            early_fault_scores=early_fault_scores,
            fault_type_scores=fault_type_scores,
            label_source_features=source_features,
        )
        spec = AnalysisSpec(
            name=str(OmegaConf.select(self.cfg, "name", default="analysis")),
            version=str(OmegaConf.select(self.cfg, "version", default="v1")),
            feature_source=str(OmegaConf.select(self.cfg, "feature_source", default="raw")),
            fit_scope=actual_fit_scope,
            enabled_sections=sections,
        ).to_dict()
        report = {
            "ok": True,
            "analysis_name": spec["name"],
            "feature_source": spec["feature_source"],
            "fit_scope": actual_fit_scope,
            "num_features": len(feature_columns(features)),
            "num_ranked_features": int(len(feature_ranking)),
            "enabled_sections": sections,
            "fault_type_skipped": bool(fault_type_skipped),
            "num_leakage_warnings": int(len(leakage_report["warnings"])),
        }
        return {
            "analysis_spec": spec,
            "analysis_report": report,
            "feature_summary": feature_summary if feature_summary is not None else pd.DataFrame(),
            "rul_correlation": rul_correlation if rul_correlation is not None else pd.DataFrame(),
            "degradation_scores": degradation_scores if degradation_scores is not None else pd.DataFrame(),
            "health_state_separability": health_state_separability if health_state_separability is not None else pd.DataFrame(),
            "early_fault_scores": early_fault_scores if early_fault_scores is not None else pd.DataFrame(),
            "fault_type_scores": fault_type_scores if fault_type_scores is not None else pd.DataFrame(),
            "feature_ranking": feature_ranking,
            "leakage_report": leakage_report,
            "figures": [],
        }

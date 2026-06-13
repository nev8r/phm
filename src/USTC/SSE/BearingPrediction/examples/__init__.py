"""
Example workflows package

this file exposes runnable notebook helper workflows

created by zyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.examples.demo_workflows import (
    create_demo_phm2012_dataset,
    create_demo_xjtu_dataset,
    run_cross_dataset_feature_export,
    run_generate_demo_datasets,
    run_phm2012_loader_overview,
    run_phm2012_mlp_feature_training,
    run_paper_cnn_lstm_attention_reproduction,
    run_xjtu_cnn_rul_training,
    run_xjtu_loader_overview,
)

__all__ = [
    "create_demo_phm2012_dataset",
    "create_demo_xjtu_dataset",
    "run_cross_dataset_feature_export",
    "run_generate_demo_datasets",
    "run_phm2012_loader_overview",
    "run_phm2012_mlp_feature_training",
    "run_paper_cnn_lstm_attention_reproduction",
    "run_xjtu_cnn_rul_training",
    "run_xjtu_loader_overview",
]

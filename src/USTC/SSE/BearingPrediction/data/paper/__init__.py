"""
paper package initialization module

this file is for exposing paper package interfaces

created by cyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.data.paper.BearingPaperDataset import (
    DEFAULT_FREQUENCY_BANDS,
    DEFAULT_SPECTRAL_FEATURES,
    DEFAULT_TIME_FEATURES,
    PHM2012_FULL_TEST_BEARINGS,
    PHM2012_LEARNING_BEARINGS,
    SequenceFeatureDataset,
    XJTU_HEALTH_STATES,
    XJTU_FAULT_TYPES,
    build_phm2012_rul_feature_cache,
    build_xjtu_binary_fault_diagnosis_feature_cache,
    build_xjtu_fault_feature_cache,
    estimate_three_sigma_fot_index,
    extract_feature_vector,
    fit_feature_standardizer,
    load_feature_cache,
    load_training_artifacts,
    make_sequence_index,
    read_phm2012_acc_file,
    read_xjtu_csv_file,
    save_training_artifacts,
    training_artifact_paths,
)

__all__ = [
    "DEFAULT_FREQUENCY_BANDS",
    "DEFAULT_SPECTRAL_FEATURES",
    "DEFAULT_TIME_FEATURES",
    "PHM2012_FULL_TEST_BEARINGS",
    "PHM2012_LEARNING_BEARINGS",
    "SequenceFeatureDataset",
    "XJTU_HEALTH_STATES",
    "XJTU_FAULT_TYPES",
    "build_phm2012_rul_feature_cache",
    "build_xjtu_binary_fault_diagnosis_feature_cache",
    "build_xjtu_fault_feature_cache",
    "estimate_three_sigma_fot_index",
    "extract_feature_vector",
    "fit_feature_standardizer",
    "load_feature_cache",
    "load_training_artifacts",
    "make_sequence_index",
    "read_phm2012_acc_file",
    "read_xjtu_csv_file",
    "save_training_artifacts",
    "training_artifact_paths",
]

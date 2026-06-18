"""
paper package initialization module

this file is for exposing paper package interfaces

created by zyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.model.paper.CNNLSTM import (
    CBAMCNNLSTMRegressor,
    CNNLSTMMultiLabelClassifier,
    PaperCBAMCNNLSTMRegressor,
    ResCNNLSTMClassifier,
)

__all__ = [
    "CBAMCNNLSTMRegressor",
    "CNNLSTMMultiLabelClassifier",
    "PaperCBAMCNNLSTMRegressor",
    "ResCNNLSTMClassifier",
]

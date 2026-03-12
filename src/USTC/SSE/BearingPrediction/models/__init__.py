"""
Models package

this file is for exposing multiple deep learning model architectures

created by zyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.models.base import MODEL_REGISTRY, BaseBearingModel
from USTC.SSE.BearingPrediction.models.cnn import CNN
from USTC.SSE.BearingPrediction.models.mlp import MLP
from USTC.SSE.BearingPrediction.models.rnn import RNN
from USTC.SSE.BearingPrediction.models.transformer import Transformer

__all__ = ["BaseBearingModel", "CNN", "MLP", "MODEL_REGISTRY", "RNN", "Transformer"]


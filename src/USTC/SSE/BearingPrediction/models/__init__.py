"""
Models package

this file is for exposing multiple deep learning model architectures

created by zyj

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.models.base import MODEL_REGISTRY, BaseBearingModel
from USTC.SSE.BearingPrediction.models.cnn import CNN
from USTC.SSE.BearingPrediction.models.cnn_lstm_attention import CNNLSTMAttention
from USTC.SSE.BearingPrediction.models.mlp import MLP
from USTC.SSE.BearingPrediction.models.rnn import RNN
from USTC.SSE.BearingPrediction.models.transformer import Transformer
from USTC.SSE.BearingPrediction.models.xlstm_transformer import FeatureSequenceTransformer, LSTMTransformer, XLSTMTransformer

__all__ = [
    "BaseBearingModel",
    "CNN",
    "CNNLSTMAttention",
    "FeatureSequenceTransformer",
    "LSTMTransformer",
    "MLP",
    "MODEL_REGISTRY",
    "RNN",
    "Transformer",
    "XLSTMTransformer",
]

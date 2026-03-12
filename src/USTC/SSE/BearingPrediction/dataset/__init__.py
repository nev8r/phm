"""
Dataset package

this file is for exposing dataset loader interfaces

created by cyy

copyright USTC

2026
"""

from USTC.SSE.BearingPrediction.dataset.base import BaseBearingLoader, DatasetResource
from USTC.SSE.BearingPrediction.dataset.loader import BearingDatasetLoader
from USTC.SSE.BearingPrediction.dataset.phm2012 import PHM2012Loader
from USTC.SSE.BearingPrediction.dataset.xjtu import XJTULoader

__all__ = ["BaseBearingLoader", "BearingDatasetLoader", "DatasetResource", "PHM2012Loader", "XJTULoader"]


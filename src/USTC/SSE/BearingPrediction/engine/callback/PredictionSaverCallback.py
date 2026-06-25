"""
Task trainer prediction saver callback placeholder.

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class PredictionSaverCallback(TrainerCallback):
    def on_train_end(self, trainer):
        return True

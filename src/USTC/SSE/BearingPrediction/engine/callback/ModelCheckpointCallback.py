"""
Task trainer checkpoint callback placeholder.

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class ModelCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, trainer):
        return True

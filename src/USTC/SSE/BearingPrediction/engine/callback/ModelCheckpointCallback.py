"""
Task trainer checkpoint callback placeholder.
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class ModelCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, trainer):
        return True

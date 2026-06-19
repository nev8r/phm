"""
Task trainer learning-rate monitor callback placeholder.
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class LearningRateMonitorCallback(TrainerCallback):
    def on_epoch_end(self, trainer):
        return True

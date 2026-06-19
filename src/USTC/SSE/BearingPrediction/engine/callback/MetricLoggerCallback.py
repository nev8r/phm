"""
Task trainer metric logger callback placeholder.
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class MetricLoggerCallback(TrainerCallback):
    def on_epoch_end(self, trainer):
        return True

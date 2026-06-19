"""
Task trainer prediction saver callback placeholder.
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class PredictionSaverCallback(TrainerCallback):
    def on_train_end(self, trainer):
        return True

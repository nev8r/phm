"""
Task trainer callback event base.

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

class TrainerCallback:
    def on_train_start(self, trainer):
        return True

    def on_epoch_start(self, trainer):
        return True

    def on_batch_end(self, trainer, batch_output):
        del batch_output
        return True

    def on_validation_end(self, trainer, metrics):
        del metrics
        return True

    def on_epoch_end(self, trainer):
        return True

    def on_train_end(self, trainer):
        return True

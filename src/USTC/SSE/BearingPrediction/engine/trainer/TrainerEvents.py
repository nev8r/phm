"""
Task trainer callback event base.
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

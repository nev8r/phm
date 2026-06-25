"""
Task trainer NaN guard callback placeholder.

Purpose: run training, testing, callbacks, metrics, or losses
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

from USTC.SSE.BearingPrediction.engine.trainer.TrainerEvents import TrainerCallback


class NaNGuardCallback(TrainerCallback):
    def on_batch_end(self, trainer, batch_output):
        if batch_output is None:
            return True
        loss = batch_output.get("loss") if isinstance(batch_output, dict) else None
        if loss is not None and not loss.isfinite().all():
            raise FloatingPointError("NaNGuardCallback detected non-finite loss")
        return True

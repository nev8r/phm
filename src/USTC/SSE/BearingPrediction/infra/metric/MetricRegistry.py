"""
Metric registry.
"""

from USTC.SSE.BearingPrediction.infra.metric.TaskMetrics import classification_metrics, regression_metrics
from USTC.SSE.BearingPrediction.infra.task.types import CLASSIFICATION_TYPES, REGRESSION


class MetricRegistry:
    @staticmethod
    def build(task_type: str):
        if task_type == REGRESSION:
            return regression_metrics
        if task_type in CLASSIFICATION_TYPES:
            return classification_metrics
        raise ValueError(f"Unsupported task_type for metrics: {task_type}")

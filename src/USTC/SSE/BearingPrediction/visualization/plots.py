"""
Visualization plot module

this file is for plotting experimental results and model attention

created by zy

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ResultVisualizer:
    """
    Plot experiment results to image files.
    """

    def __init__(self, style: str = "whitegrid", dpi: int = 150) -> None:
        self.style = style
        self.dpi = dpi
        sns.set_theme(style=style)

    def plot_prediction_curve(self, targets: np.ndarray, predictions: np.ndarray, output_path: Path, uncertainties: np.ndarray | None = None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 5), dpi=self.dpi)
        axis.plot(targets, label="Target", color="#1d3557")
        axis.plot(predictions, label="Prediction", color="#e76f51")
        if uncertainties is not None:
            lower_bound = predictions - uncertainties
            upper_bound = predictions + uncertainties
            axis.fill_between(np.arange(predictions.size), lower_bound, upper_bound, alpha=0.2, color="#f4a261", label="Uncertainty")
        axis.set_title("Prediction Curve")
        axis.set_xlabel("Sample Index")
        axis.set_ylabel("Value")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def plot_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray, output_path: Path, labels: list[str] | None = None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(targets, predictions)
        figure, axis = plt.subplots(figsize=(6, 5), dpi=self.dpi)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis, xticklabels=labels, yticklabels=labels)
        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def plot_degradation_stages(self, stage_frame: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 5), dpi=self.dpi)
        sns.lineplot(data=stage_frame, x="sample_index", y="health_indicator", hue="stage_name", palette="viridis", ax=axis)
        axis.set_title("Degradation Stage Partition")
        axis.set_xlabel("Sample Index")
        axis.set_ylabel("Health Indicator")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def plot_attention_heatmap(self, attention_weights: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_values = attention_weights
        if heatmap_values.ndim == 4:
            heatmap_values = heatmap_values.mean(axis=(0, 1))
        elif heatmap_values.ndim == 3:
            heatmap_values = heatmap_values.mean(axis=0)
        figure, axis = plt.subplots(figsize=(6, 5), dpi=self.dpi)
        sns.heatmap(heatmap_values, cmap="magma", ax=axis)
        axis.set_title("Attention Heatmap")
        axis.set_xlabel("Key Position")
        axis.set_ylabel("Query Position")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)


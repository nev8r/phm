"""
Visualization module

this file is for plotting bearing health and prediction results

created by zy

copyright USTC

2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from USTC.SSE.BearingPrediction.config import VisualizationConfig


class BearingVisualizationService:
    """
    Create project visualizations and save them as image files.
    """

    def __init__(self, visualization_config: VisualizationConfig) -> None:
        self.visualization_config = visualization_config
        sns.set_theme(style=visualization_config.style)

    def plot_health_trend(self, dataset: pd.DataFrame, output_path: Path) -> None:
        """
        plot equipment health trend across lifecycle

        Parameters
        ----------
        dataset : pd.DataFrame
            cycle level dataset
        output_path : Path
            image output path
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_frame = (
            dataset.groupby("cycle", as_index=False)[["health_index", "temperature"]]
            .mean()
            .sort_values("cycle")
        )
        temperature_min = float(summary_frame["temperature"].min())
        temperature_range = float(summary_frame["temperature"].max() - temperature_min)
        summary_frame["temperature_scaled"] = (
            (summary_frame["temperature"] - temperature_min) / temperature_range if temperature_range else 0.0
        )

        figure, axis = plt.subplots(figsize=(10, 5), dpi=self.visualization_config.figure_dpi)
        sns.lineplot(data=summary_frame, x="cycle", y="health_index", ax=axis, label="Health Index", color="#0b6e4f")
        sns.lineplot(data=summary_frame, x="cycle", y="temperature_scaled", ax=axis, label="Temperature", color="#cc5803")
        axis.set_title("Bearing Health Trend")
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Normalized Indicator")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def plot_rul_predictions(self, prediction_frame: pd.DataFrame, output_path: Path) -> None:
        """
        plot actual and predicted rul values

        Parameters
        ----------
        prediction_frame : pd.DataFrame
            prediction result table
        output_path : Path
            image output path
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_frame = prediction_frame.head(120).copy().reset_index(drop=True)
        plot_frame["sample_index"] = plot_frame.index

        figure, axis = plt.subplots(figsize=(10, 5), dpi=self.visualization_config.figure_dpi)
        sns.lineplot(data=plot_frame, x="sample_index", y="rul", ax=axis, label="Actual RUL", color="#004e98")
        sns.lineplot(
            data=plot_frame,
            x="sample_index",
            y="predicted_rul",
            ax=axis,
            label="Predicted RUL",
            color="#ff6700",
        )
        axis.set_title("RUL Prediction Curve")
        axis.set_xlabel("Sample Index")
        axis.set_ylabel("RUL")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

    def plot_survival_curve(self, survival_curve: pd.DataFrame, output_path: Path) -> None:
        """
        plot survival probabilities over the prediction horizon

        Parameters
        ----------
        survival_curve : pd.DataFrame
            survival curve result
        output_path : Path
            image output path
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 5), dpi=self.visualization_config.figure_dpi)
        sns.lineplot(
            data=survival_curve,
            x="timeline",
            y="kaplan_meier_survival_probability",
            ax=axis,
            label="Kaplan-Meier",
            color="#6a4c93",
        )
        sns.lineplot(
            data=survival_curve,
            x="timeline",
            y="cox_survival_probability",
            ax=axis,
            label="Cox Model",
            color="#1982c4",
        )
        axis.set_title("Failure Survival Probability")
        axis.set_xlabel("Future Cycles")
        axis.set_ylabel("Survival Probability")
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)

"""
Serialization module

this file is for importing and exporting data, cache, model and result artifacts

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class ArtifactSerializer:
    """
    Serialize tabular or generic artifacts to multiple file formats.
    """

    @staticmethod
    def supports_yaml() -> bool:
        """
        report whether yaml serialization is available

        Returns
        -------
        bool
            True when PyYAML is installed
        """

        return yaml is not None

    @staticmethod
    def save_dataframe(data_frame: pd.DataFrame, output_path: Path) -> None:
        """
        save dataframe to csv or pickle

        Parameters
        ----------
        data_frame : pd.DataFrame
            dataframe to save
        output_path : Path
            target path
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            data_frame.to_csv(output_path, index=False)
            return
        if output_path.suffix.lower() in {".pkl", ".pickle"}:
            data_frame.to_pickle(output_path)
            return
        raise ValueError(f"unsupported dataframe output format: {output_path.suffix}")

    @staticmethod
    def load_dataframe(input_path: Path) -> pd.DataFrame:
        """
        load dataframe from csv or pickle

        Parameters
        ----------
        input_path : Path
            source path

        Returns
        -------
        pd.DataFrame
            loaded dataframe
        """

        if input_path.suffix.lower() == ".csv":
            return pd.read_csv(input_path)
        if input_path.suffix.lower() in {".pkl", ".pickle"}:
            return pd.read_pickle(input_path)
        raise ValueError(f"unsupported dataframe input format: {input_path.suffix}")

    @staticmethod
    def save_object(data_object: Any, output_path: Path) -> None:
        """
        save object to json, yaml or pickle

        Parameters
        ----------
        data_object : Any
            object to save
        output_path : Path
            target path
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(data_object, output_file, indent=2, ensure_ascii=False)
            return
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("yaml serialization requires the optional dependency PyYAML")
            with output_path.open("w", encoding="utf-8") as output_file:
                yaml.safe_dump(data_object, output_file, sort_keys=False, allow_unicode=True)
            return
        if suffix in {".pkl", ".pickle"}:
            with output_path.open("wb") as output_file:
                pickle.dump(data_object, output_file)
            return
        raise ValueError(f"unsupported object output format: {output_path.suffix}")

    @staticmethod
    def load_object(input_path: Path) -> Any:
        """
        load object from json, yaml or pickle

        Parameters
        ----------
        input_path : Path
            source path

        Returns
        -------
        Any
            loaded object
        """

        suffix = input_path.suffix.lower()
        if suffix == ".json":
            with input_path.open("r", encoding="utf-8") as input_file:
                return json.load(input_file)
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("yaml deserialization requires the optional dependency PyYAML")
            with input_path.open("r", encoding="utf-8") as input_file:
                return yaml.safe_load(input_file)
        if suffix in {".pkl", ".pickle"}:
            with input_path.open("rb") as input_file:
                return pickle.load(input_file)
        raise ValueError(f"unsupported object input format: {input_path.suffix}")


class ModelIO:
    """
    Save and load PyTorch models.
    """

    @staticmethod
    def save_checkpoint(model: torch.nn.Module, output_path: Path, metadata: dict[str, Any] | None = None) -> None:
        """
        save model state dict and metadata

        Parameters
        ----------
        model : torch.nn.Module
            model to save
        output_path : Path
            output checkpoint path
        metadata : dict[str, Any] | None
            checkpoint metadata
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, output_path)

    @staticmethod
    def load_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
        """
        load checkpoint into model

        Parameters
        ----------
        model : torch.nn.Module
            model to restore
        checkpoint_path : Path
            checkpoint path

        Returns
        -------
        dict[str, Any]
            checkpoint metadata
        """

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        return checkpoint.get("metadata", {})


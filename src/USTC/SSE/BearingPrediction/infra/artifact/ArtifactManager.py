"""
Artifact manager module.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import json
from pathlib import Path
from typing import Any, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


class ArtifactManager:
    """
    Manage paths and simple artifact serialization under one root directory.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: Union[str, Path]) -> Path:
        return self.root / Path(relative_path)

    def mkdir(self, relative_path: Union[str, Path]) -> Path:
        path = self.path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: Union[str, Path], data: Any) -> Path:
        path = self._prepare_write(relative_path)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=self._json_default) + "\n",
            encoding="utf-8",
        )
        return path

    def read_json(self, relative_path: Union[str, Path]) -> Any:
        return json.loads(self.path(relative_path).read_text(encoding="utf-8"))

    def write_text(self, relative_path: Union[str, Path], text: str) -> Path:
        path = self._prepare_write(relative_path)
        path.write_text(text, encoding="utf-8")
        return path

    def write_yaml(self, relative_path: Union[str, Path], data: Any) -> Path:
        path = self._prepare_write(relative_path)
        config = data if isinstance(data, (DictConfig, ListConfig)) else OmegaConf.create(data)
        OmegaConf.save(config=config, f=path, resolve=True)
        return path

    def _prepare_write(self, relative_path: Union[str, Path]) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        return str(value)

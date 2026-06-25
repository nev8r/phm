"""
Run context module.

Purpose: provide infrastructure services for indexed, configurable experiments
Author: zdh
Program date: 2026-06
Copyright: USTC

2026
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.artifact.ArtifactManager import ArtifactManager


@dataclass
class RunContext:
    """
    Metadata and artifact paths for a single run.
    """

    run_id: str
    run_name: str
    project_name: str
    artifact_root: Path
    run_dir: Path
    seed: int
    created_at: str
    tags: List[str] = field(default_factory=list)
    artifacts: ArtifactManager = field(repr=False, compare=False, default=None)

    @classmethod
    def create(cls, cfg: DictConfig) -> "RunContext":
        project_name = str(OmegaConf.select(cfg, "project.name"))
        run_name = str(OmegaConf.select(cfg, "run.name", default="run"))
        artifact_root = Path(str(OmegaConf.select(cfg, "project.artifact_root"))).expanduser()
        seed = int(OmegaConf.select(cfg, "project.seed"))
        tags = list(OmegaConf.select(cfg, "run.tags", default=[]))

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_{_slugify(run_name)}_{uuid.uuid4().hex[:8]}"
        run_dir = artifact_root / "runs" / run_id
        artifacts = ArtifactManager(run_dir)

        return cls(
            run_id=run_id,
            run_name=run_name,
            project_name=project_name,
            artifact_root=artifact_root,
            run_dir=run_dir,
            seed=seed,
            created_at=created_at,
            tags=tags,
            artifacts=artifacts,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "project_name": self.project_name,
            "artifact_root": str(self.artifact_root),
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    def save_metadata(self) -> Path:
        return self.artifacts.write_json("run.json", self.to_dict())

    def save_resolved_config(self, cfg: DictConfig) -> Path:
        return self.artifacts.write_yaml("config/resolved.yaml", cfg)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return slug or "run"

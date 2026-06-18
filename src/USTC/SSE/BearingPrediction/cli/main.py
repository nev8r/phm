"""
Experiment command line entrypoint.

This module provides the Hydra-based CLI used to validate experiment config,
create run artifacts, and dispatch early infrastructure stages.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from USTC.SSE.BearingPrediction.infra.experiment.RunContext import RunContext


ALLOWED_MODES = {
    "validate",
    "build_index",
    "extract_features",
    "build_labels",
    "analyze_features",
    "inspect_task",
    "train",
    "eval",
    "run",
}

STAGE_FOR_MODE = {
    "build_index": "Stage 1",
    "extract_features": "Stage 2",
    "build_labels": "Stage 3",
    "inspect_task": "Stage 4",
    "train": "Stage 5",
    "eval": "Stage 5",
    "run": "Stage 5",
    "analyze_features": "Stage 7",
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    config_name, overrides = parse_cli_args(sys.argv[1:] if argv is None else argv)
    conf_dir = find_conf_dir()

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    run_validate_cli(cfg)


def run_validate_cli(cfg: DictConfig) -> None:
    context = RunContext.create(cfg)
    context.save_resolved_config(cfg)
    context.save_metadata()

    report = validate_config(cfg, context)
    context.artifacts.write_json("validation_report.json", report)

    if not report["ok"]:
        failed = ", ".join(check["name"] for check in report["checks"] if not check["ok"])
        raise ValueError(f"Config validation failed: {failed}")

    mode = str(cfg.mode)
    if mode == "validate":
        print(f"Validation succeeded. Run directory: {context.run_dir}")
        return

    if mode == "build_index":
        run_build_index(cfg, context)
        return

    stage = STAGE_FOR_MODE.get(mode, "a later stage")
    raise NotImplementedError(f"mode={mode} will be implemented in {stage}")


def run_build_index(cfg: DictConfig, context: RunContext) -> None:
    from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
    from USTC.SSE.BearingPrediction.infra.index.IndexValidator import IndexValidator
    from USTC.SSE.BearingPrediction.infra.split.SplitRegistry import build_splitter

    index = IndexBuilder().build(cfg)
    index_report = IndexValidator().validate(index, strict=True)
    context.artifacts.mkdir("index")
    index.to_parquet(context.artifacts.path("index/sample_index.parquet"), index=False)
    index.to_csv(context.artifacts.path("index/sample_index.csv"), index=False)
    context.artifacts.write_json("index/index_report.json", index_report)

    if bool(OmegaConf.select(cfg, "split.enabled", default=False)):
        splitter = build_splitter(cfg.split)
        split = splitter.split(index)
        context.artifacts.mkdir("split")
        context.artifacts.write_json("split/split.json", split.to_dict())
        context.artifacts.write_json("split/split_report.json", split.report())

    print(f"Index build succeeded. Run directory: {context.run_dir}")


def parse_cli_args(argv: Sequence[str]) -> Tuple[str, List[str]]:
    config_name = "smoke"
    overrides: List[str] = []
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--config-name":
            if index + 1 >= len(argv):
                raise ValueError("--config-name requires a value")
            config_name = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            index += 1
            continue
        if arg.startswith("hydra."):
            index += 1
            continue
        if arg.startswith("--"):
            raise ValueError(f"Unsupported CLI option: {arg}")

        overrides.append(arg)
        index += 1

    return config_name, overrides


def find_conf_dir() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for base in candidates:
        conf_dir = base / "conf"
        if (conf_dir / "smoke.yaml").is_file():
            return conf_dir
    raise FileNotFoundError("Could not locate repository conf/ directory")


def validate_config(cfg: DictConfig, context: RunContext) -> Dict[str, Any]:
    mode = str(OmegaConf.select(cfg, "mode", default=""))
    dataset_name = str(OmegaConf.select(cfg, "dataset.name", default=""))
    checks: List[Dict[str, Any]] = [
        _check("project.name", _is_non_empty_string(OmegaConf.select(cfg, "project.name", default=None))),
        _check("project.seed", isinstance(OmegaConf.select(cfg, "project.seed", default=None), int)),
        _check("artifact_root", context.artifact_root.exists() and context.artifact_root.is_dir()),
        _check("dataset.name", _is_non_empty_string(dataset_name)),
        _check("mode", mode in ALLOWED_MODES),
    ]

    return {
        "ok": all(check["ok"] for check in checks),
        "mode": mode,
        "dataset": dataset_name,
        "run_id": context.run_id,
        "run_dir": str(context.run_dir),
        "checks": checks,
    }


def _check(name: str, ok: bool) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok)}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


if __name__ == "__main__":
    main()

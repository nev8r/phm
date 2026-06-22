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
import pandas as pd

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
    "analyze_features": "Stage 6",
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

    if mode == "extract_features":
        run_extract_features(cfg, context)
        return

    if mode == "build_labels":
        run_build_labels(cfg, context)
        return

    if mode == "analyze_features":
        run_analyze_features(cfg, context)
        return

    if mode == "inspect_task":
        run_inspect_task(cfg, context)
        return

    if mode == "train":
        run_train(cfg, context)
        return

    if mode == "eval":
        run_eval(cfg, context)
        return

    if mode == "run":
        run_train(cfg, context)
        return

    stage = STAGE_FOR_MODE.get(mode, "a later stage")
    raise NotImplementedError(f"mode={mode} will be implemented in {stage}")


def run_build_index(cfg: DictConfig, context: RunContext) -> None:
    build_index_artifacts(cfg, context)
    print(f"Index build succeeded. Run directory: {context.run_dir}")


def run_extract_features(cfg: DictConfig, context: RunContext) -> None:
    index, split = build_index_artifacts(cfg, context)
    build_feature_artifacts(cfg, context, index, split)
    print(f"Feature extraction succeeded. Run directory: {context.run_dir}")


def run_build_labels(cfg: DictConfig, context: RunContext) -> None:
    index, split = build_index_artifacts(cfg, context)
    raw_features = None
    cleaned_features = None
    if bool(OmegaConf.select(cfg, "label.requires_features", default=False)):
        raw_features, cleaned_features, _, _ = build_feature_artifacts(cfg, context, index, split)

    build_label_artifacts(cfg, context, index, split, raw_features, cleaned_features)
    print(f"Label build succeeded. Run directory: {context.run_dir}")


def run_analyze_features(cfg: DictConfig, context: RunContext) -> None:
    from USTC.SSE.BearingPrediction.analysis.AnalysisBuilder import AnalysisBuilder
    from USTC.SSE.BearingPrediction.analysis.AnalysisStore import AnalysisStore

    index, split = build_index_artifacts(cfg, context)
    if not bool(OmegaConf.select(cfg, "feature.enabled", default=False)):
        raise ValueError("mode=analyze_features requires an enabled feature config")
    if not bool(OmegaConf.select(cfg, "label.enabled", default=False)):
        raise ValueError("mode=analyze_features requires an enabled label config")

    raw_features, cleaned_features, _, _ = build_feature_artifacts(cfg, context, index, split)
    labels, _, _, hi, fpt = build_label_artifacts(cfg, context, index, split, raw_features, cleaned_features)
    feature_source = str(OmegaConf.select(cfg, "analysis.feature_source", default="raw"))
    if feature_source == "raw":
        features_for_analysis = raw_features
    elif feature_source == "cleaned":
        features_for_analysis = cleaned_features
    else:
        raise ValueError(f"Unsupported analysis.feature_source: {feature_source}")

    outputs = AnalysisBuilder(cfg.analysis).build(
        features=features_for_analysis,
        labels=labels,
        index=index,
        split_result=split,
        hi=hi,
        fpt=fpt,
    )
    store = AnalysisStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "analysis.store.write_csv", default=True)),
        write_figures=bool(OmegaConf.select(cfg, "analysis.store.write_figures", default=True)),
    )
    store.save(outputs)
    print(f"Feature analysis succeeded. Run directory: {context.run_dir}")


def run_inspect_task(cfg: DictConfig, context: RunContext) -> None:
    from USTC.SSE.BearingPrediction.infra.task.TaskStore import TaskStore

    datamodule = build_task_datamodule_artifacts(cfg, context)
    store = TaskStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "task.store.write_csv", default=True)),
    )
    store.save(
        manifest=datamodule.task_manifest,
        task_spec=datamodule.task_spec,
        task_report=datamodule.task_report,
        feature_columns=datamodule.feature_columns,
        target_columns=datamodule.target_columns,
    )
    print(f"Task inspection succeeded. Run directory: {context.run_dir}")


def run_train(cfg: DictConfig, context: RunContext) -> None:
    from USTC.SSE.BearingPrediction.engine.trainer.ConfigurableTrainer import ConfigurableTrainer
    from USTC.SSE.BearingPrediction.infra.model.ModelFactory import ModelFactory
    from USTC.SSE.BearingPrediction.infra.task.TaskStore import TaskStore

    datamodule = build_task_datamodule_artifacts(cfg, context)
    TaskStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "task.store.write_csv", default=True)),
    ).save(
        manifest=datamodule.task_manifest,
        task_spec=datamodule.task_spec,
        task_report=datamodule.task_report,
        feature_columns=datamodule.feature_columns,
        target_columns=datamodule.target_columns,
    )
    model, model_spec = ModelFactory(cfg.model).build(datamodule=datamodule, task_cfg=cfg.task)
    ConfigurableTrainer(cfg, context, datamodule, model, model_spec).train()
    print(f"Training succeeded. Run directory: {context.run_dir}")


def run_eval(cfg: DictConfig, context: RunContext) -> None:
    from USTC.SSE.BearingPrediction.engine.trainer.ConfigurableTrainer import ConfigurableTrainer
    from USTC.SSE.BearingPrediction.infra.model.ModelFactory import ModelFactory
    from USTC.SSE.BearingPrediction.infra.task.TaskStore import TaskStore

    checkpoint = OmegaConf.select(cfg, "checkpoint", default=None)
    if checkpoint in (None, "null", ""):
        raise ValueError("mode=eval requires checkpoint=/path/to/checkpoint")
    datamodule = build_task_datamodule_artifacts(cfg, context)
    TaskStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "task.store.write_csv", default=True)),
    ).save(
        manifest=datamodule.task_manifest,
        task_spec=datamodule.task_spec,
        task_report=datamodule.task_report,
        feature_columns=datamodule.feature_columns,
        target_columns=datamodule.target_columns,
    )
    model, model_spec = ModelFactory(cfg.model).build(datamodule=datamodule, task_cfg=cfg.task)
    ConfigurableTrainer(cfg, context, datamodule, model, model_spec).evaluate_checkpoint(str(checkpoint))
    print(f"Evaluation succeeded. Run directory: {context.run_dir}")


def build_task_datamodule_artifacts(cfg: DictConfig, context: RunContext):
    from USTC.SSE.BearingPrediction.infra.task.TaskBuilder import TaskBuilder

    cache_dir = OmegaConf.select(cfg, "task.cache_dir", default=None)
    if cache_dir not in (None, "null", ""):
        return _load_task_datamodule_from_cache(Path(str(cache_dir)).expanduser())

    index, split = build_index_artifacts(cfg, context)
    if not bool(OmegaConf.select(cfg, "feature.enabled", default=False)):
        raise ValueError("task modes require an enabled feature config")
    raw_features, cleaned_features, _, _ = build_feature_artifacts(cfg, context, index, split)
    labels, _, _, _, _ = build_label_artifacts(cfg, context, index, split, raw_features, cleaned_features)
    feature_source = str(OmegaConf.select(cfg, "task.feature_source", default="cleaned"))
    if feature_source == "cleaned":
        features_for_task = cleaned_features
    elif feature_source == "raw":
        features_for_task = raw_features
    else:
        raise ValueError(f"Unsupported task.feature_source: {feature_source}")
    return TaskBuilder(cfg.task).build(
        features=features_for_task,
        labels=labels,
        split_result=split,
    )


def _load_task_datamodule_from_cache(cache_dir: Path):
    from USTC.SSE.BearingPrediction.infra.task.DataModule import DataModule
    from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset

    if not cache_dir.exists():
        raise FileNotFoundError(f"Task cache directory does not exist: {cache_dir}")
    task_dir = cache_dir / "task"
    feature_dir = cache_dir / "features"
    label_dir = cache_dir / "labels"
    required = [
        task_dir / "task_manifest.parquet",
        task_dir / "task_spec.json",
        task_dir / "task_report.json",
        task_dir / "feature_columns.txt",
        task_dir / "target_columns.txt",
        label_dir / "labels.parquet",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Task cache is missing required artifact: {path}")

    task_spec = OmegaConf.to_container(OmegaConf.load(task_dir / "task_spec.json"), resolve=True)
    task_report = OmegaConf.to_container(OmegaConf.load(task_dir / "task_report.json"), resolve=True)
    feature_source = str(task_spec.get("feature_source", "cleaned"))
    feature_file = feature_dir / ("cleaned_features.parquet" if feature_source == "cleaned" else "raw_features.parquet")
    if not feature_file.exists():
        raise FileNotFoundError(f"Task cache is missing feature artifact: {feature_file}")

    features = pd.read_parquet(feature_file)
    labels = pd.read_parquet(label_dir / "labels.parquet")
    manifest = pd.read_parquet(task_dir / "task_manifest.parquet")
    feature_columns = _read_lines(task_dir / "feature_columns.txt")
    target_columns = _read_lines(task_dir / "target_columns.txt")
    task_type = str(task_spec["task_type"])
    input_mode = str(task_spec["input_mode"])

    def dataset_for(split_name: str):
        subset = manifest[manifest["split"] == split_name].reset_index(drop=True)
        if subset.empty:
            return None
        return TaskDataset(features, labels, subset, feature_columns, target_columns, input_mode, task_type)

    print(f"Loaded task cache: {cache_dir}", flush=True)
    return DataModule(
        train=dataset_for("train"),
        val=dataset_for("val"),
        test=dataset_for("test"),
        all=dataset_for("all"),
        task_manifest=manifest,
        feature_columns=feature_columns,
        target_columns=target_columns,
        task_spec=task_spec,
        task_report=task_report,
    )


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_label_artifacts(cfg: DictConfig, context: RunContext, index, split, raw_features=None, cleaned_features=None):
    from USTC.SSE.BearingPrediction.infra.label.LabelBuilder import LabelBuilder
    from USTC.SSE.BearingPrediction.infra.label.LabelStore import LabelStore

    labels, label_spec, label_report, hi, fpt = LabelBuilder(cfg.label).build(
        index=index,
        raw_features=raw_features,
        cleaned_features=cleaned_features,
        split=split,
    )
    store = LabelStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "label.store.write_csv", default=True)),
    )
    store.save(labels, label_spec, label_report, hi=hi, fpt=fpt)
    return labels, label_spec, label_report, hi, fpt


def build_feature_artifacts(cfg: DictConfig, context: RunContext, index, split):
    from USTC.SSE.BearingPrediction.infra.feature.FeatureCleaner import FeatureCleaner
    from USTC.SSE.BearingPrediction.infra.feature.FeatureExtractor import FeatureExtractor
    from USTC.SSE.BearingPrediction.infra.feature.FeatureReport import build_feature_report
    from USTC.SSE.BearingPrediction.infra.feature.FeatureStore import FeatureStore

    raw_features, feature_spec, backend_reports = FeatureExtractor(cfg.feature).extract(index)
    cleaner = FeatureCleaner(cfg.feature.cleaner)
    train_sample_uids = split.train_sample_uids if split is not None else None
    cleaned_features = cleaner.fit_transform(raw_features, train_sample_uids=train_sample_uids)
    cleaner_fit_scope = "train_only" if split is not None else "all_no_split"
    feature_report = build_feature_report(
        raw_features=raw_features,
        cleaned_features=cleaned_features,
        feature_set=str(cfg.feature.name),
        backend_reports=backend_reports,
        cleaner=cleaner,
        cleaner_fit_scope=cleaner_fit_scope,
    )
    store = FeatureStore(
        context.artifacts,
        write_csv=bool(OmegaConf.select(cfg, "feature.store.write_csv", default=True)),
    )
    store.save(raw_features, cleaned_features, feature_spec, feature_report, cleaner.state_dict())
    return raw_features, cleaned_features, feature_spec, feature_report


def build_index_artifacts(cfg: DictConfig, context: RunContext):
    from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
    from USTC.SSE.BearingPrediction.infra.index.IndexValidator import IndexValidator
    from USTC.SSE.BearingPrediction.infra.split.SplitRegistry import build_splitter

    index = IndexBuilder().build(cfg)
    index_report = IndexValidator().validate(index, strict=True)
    context.artifacts.mkdir("index")
    index.to_parquet(context.artifacts.path("index/sample_index.parquet"), index=False)
    index.to_csv(context.artifacts.path("index/sample_index.csv"), index=False)
    context.artifacts.write_json("index/index_report.json", index_report)

    split = None
    if bool(OmegaConf.select(cfg, "split.enabled", default=False)):
        splitter = build_splitter(cfg.split)
        split = splitter.split(index)
        context.artifacts.mkdir("split")
        context.artifacts.write_json("split/split.json", split.to_dict())
        context.artifacts.write_json("split/split_report.json", split.report())
    return index, split


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

"""
Standalone sklearn/XGBoost tabular baseline recipe.

This recipe intentionally stays outside the torch ModelFactory and trainer. It
reuses the existing PHM data, feature, label, split, and task builders, then
fits a scikit-learn-compatible tabular model on the constructed TaskDataset.
"""

from __future__ import annotations

import json
import math
import pickle
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from USTC.SSE.BearingPrediction.cli.main import find_conf_dir, parse_cli_args
from USTC.SSE.BearingPrediction.infra.experiment.RunContext import RunContext
from USTC.SSE.BearingPrediction.infra.feature.FeatureCleaner import FeatureCleaner
from USTC.SSE.BearingPrediction.infra.feature.FeatureExtractor import FeatureExtractor
from USTC.SSE.BearingPrediction.infra.index.IndexBuilder import IndexBuilder
from USTC.SSE.BearingPrediction.infra.label.LabelBuilder import LabelBuilder
from USTC.SSE.BearingPrediction.infra.split.SplitRegistry import build_splitter
from USTC.SSE.BearingPrediction.infra.task.DataModule import DataModule
from USTC.SSE.BearingPrediction.infra.task.TaskDataset import TaskDataset
from USTC.SSE.BearingPrediction.infra.task.TaskBuilder import TaskBuilder
from USTC.SSE.BearingPrediction.infra.task.types import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TYPES,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)


CURATED_FILES = [
    "command.txt",
    "config.json",
    "task_spec.json",
    "task_report.json",
    "feature_columns.txt",
    "target_columns.txt",
    "metrics.json",
    "feature_importance.csv",
    "experiment_report.md",
]


@dataclass
class DatasetArrays:
    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "uv run python " + " ".join(shlex.quote(part) for part in sys.argv)
    cfg = compose_recipe_config(argv)
    context = RunContext.create(cfg)
    context.save_metadata()

    datamodule = build_datamodule(cfg)
    result = fit_and_save(cfg, context, datamodule, command)
    copy_curated_outputs(context.run_dir, result["curated_dir"])
    print(f"Non-MLP baseline completed: {result['experiment_id']}")
    print(f"Raw run directory: {context.run_dir}")
    print(f"Curated directory: {result['curated_dir']}")


def compose_recipe_config(argv: Sequence[str]) -> DictConfig:
    config_name, overrides = parse_cli_args(argv)
    sklearn_overrides, hydra_overrides = _split_sklearn_overrides(overrides)

    with initialize_config_dir(version_base=None, config_dir=str(find_conf_dir())):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)

    for override in sklearn_overrides:
        key, value = override.split("=", 1)
        parsed = OmegaConf.from_dotlist([override])
        OmegaConf.update(cfg, key, OmegaConf.select(parsed, key), merge=True, force_add=True)

    if OmegaConf.select(cfg, "sklearn.model", default=None) is None:
        raise ValueError("sklearn.model is required")
    if OmegaConf.select(cfg, "sklearn.output_root", default=None) is None:
        OmegaConf.update(
            cfg,
            "sklearn.output_root",
            "reports/non_mlp_baseline_results",
            merge=True,
            force_add=True,
        )
    return cfg


def build_datamodule(cfg: DictConfig) -> DataModule:
    index = IndexBuilder().build(cfg)
    split = None
    if bool(OmegaConf.select(cfg, "split.enabled", default=False)):
        split = build_splitter(cfg.split).split(index)

    raw_features, _, _ = FeatureExtractor(cfg.feature).extract(index)
    train_sample_uids = split.train_sample_uids if split is not None else None
    cleaned_features = FeatureCleaner(cfg.feature.cleaner).fit_transform(raw_features, train_sample_uids=train_sample_uids)
    labels, _, _, _, _ = LabelBuilder(cfg.label).build(
        index=index,
        raw_features=raw_features,
        cleaned_features=cleaned_features,
        split=split,
    )

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


def build_model(model_name: str, task_type: str, random_state: int = 42):
    if model_name == "xgboost_regressor":
        _require_task_type(model_name, task_type, {REGRESSION})
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost_classifier":
        _require_task_type(model_name, task_type, CLASSIFICATION_TYPES)
        from xgboost import XGBClassifier

        objective = "binary:logistic" if task_type == BINARY_CLASSIFICATION else "multi:softprob"
        eval_metric = "logloss" if task_type == BINARY_CLASSIFICATION else "mlogloss"
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective=objective,
            eval_metric=eval_metric,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "random_forest_regressor":
        _require_task_type(model_name, task_type, {REGRESSION})
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "random_forest_classifier":
        _require_task_type(model_name, task_type, CLASSIFICATION_TYPES)
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported sklearn.model: {model_name}")


def dataset_to_arrays(dataset: TaskDataset) -> DatasetArrays:
    sample_uids = dataset.manifest["target_sample_uid"].astype(str).tolist()
    x = dataset.features.loc[sample_uids, dataset.feature_columns].to_numpy(dtype=np.float32)
    y_frame = dataset.labels.loc[sample_uids, dataset.target_columns]
    if dataset.task_type in CLASSIFICATION_TYPES:
        y = y_frame.iloc[:, 0].to_numpy(dtype=np.int64)
    elif dataset.task_type == REGRESSION:
        y = y_frame.to_numpy(dtype=np.float32)
        if y.shape[1] == 1:
            y = y[:, 0]
    else:
        raise ValueError(f"Unsupported task_type: {dataset.task_type}")

    metadata = dataset.manifest[[
        "example_uid",
        "split",
        "dataset",
        "bearing_id",
        "condition_id",
        "source_group",
        "target_sample_uid",
        "target_timestep",
    ]].copy()
    metadata = metadata.rename(columns={"target_sample_uid": "sample_uid"})
    return DatasetArrays(x=x, y=y, metadata=metadata)


def compute_metrics(task_type: str, y_true: Iterable, y_pred: Iterable) -> Dict[str, Any]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if task_type == REGRESSION:
        mae = float(mean_absolute_error(y_true_array, y_pred_array))
        mse = float(mean_squared_error(y_true_array, y_pred_array))
        rmse = float(math.sqrt(mse))
        return {
            "primary_metric": "RMSE",
            "metric_direction": "lower_is_better",
            "metrics": {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
            },
        }
    if task_type in CLASSIFICATION_TYPES:
        return {
            "primary_metric": "WeightedF1",
            "metric_direction": "higher_is_better",
            "metrics": {
                "Accuracy": float(accuracy_score(y_true_array, y_pred_array)),
                "MacroF1": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
                "WeightedF1": float(f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
            },
        }
    raise ValueError(f"Unsupported task_type: {task_type}")


def feature_importance_frame(feature_columns: List[str], importances: Iterable[float]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "feature": feature_columns,
        "importance": [float(value) for value in importances],
    })
    frame = frame.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame[["rank", "feature", "importance"]]


def fit_and_save(cfg: DictConfig, context: RunContext, datamodule: DataModule, command: str) -> Dict[str, Any]:
    if datamodule.train is None or datamodule.val is None or datamodule.test is None:
        raise ValueError("Step Y requires non-empty train, val, and test splits")

    model_name = str(OmegaConf.select(cfg, "sklearn.model"))
    task_type = str(datamodule.task_spec["task_type"])
    seed = int(OmegaConf.select(cfg, "project.seed", default=42))
    model = build_model(model_name, task_type, random_state=seed)

    train_arrays = dataset_to_arrays(datamodule.train)
    val_arrays = dataset_to_arrays(datamodule.val)
    test_arrays = dataset_to_arrays(datamodule.test)

    model.fit(train_arrays.x, train_arrays.y)
    val_pred = model.predict(val_arrays.x)
    test_pred = model.predict(test_arrays.x)
    val_result = compute_metrics(task_type, val_arrays.y, val_pred)
    test_result = compute_metrics(task_type, test_arrays.y, test_pred)
    primary_metric = test_result["primary_metric"]

    metrics_payload = {
        "experiment_id": context.run_name,
        "run_id": context.run_id,
        "fit_status": "completed",
        "dataset": str(OmegaConf.select(cfg, "dataset.name", default="")),
        "split": str(OmegaConf.select(cfg, "split.name", default="")),
        "task": str(OmegaConf.select(cfg, "task.name", default="")),
        "task_type": task_type,
        "model": model_name,
        "feature_count": len(datamodule.feature_columns),
        "target_columns": datamodule.target_columns,
        "train_examples": int(len(train_arrays.y)),
        "val_examples": int(len(val_arrays.y)),
        "test_examples": int(len(test_arrays.y)),
        "primary_metric": primary_metric,
        "metric_direction": test_result["metric_direction"],
        "val_metrics": val_result["metrics"],
        "test_metrics": test_result["metrics"],
        "val_primary": float(val_result["metrics"][primary_metric]),
        "test_primary": float(test_result["metrics"][primary_metric]),
    }

    importances = getattr(model, "feature_importances_", np.zeros(len(datamodule.feature_columns), dtype=float))
    importance = feature_importance_frame(datamodule.feature_columns, importances)
    curated_dir = Path(str(OmegaConf.select(cfg, "sklearn.output_root"))) / context.run_name

    _write_text(context.run_dir / "command.txt", command + "\n")
    _write_json(context.run_dir / "config.json", OmegaConf.to_container(cfg, resolve=True))
    _write_json(context.run_dir / "run.json", {**context.to_dict(), "fit_status": "completed"})
    _write_json(context.run_dir / "task_spec.json", datamodule.task_spec)
    _write_json(context.run_dir / "task_report.json", datamodule.task_report)
    _write_text(context.run_dir / "feature_columns.txt", "\n".join(datamodule.feature_columns) + "\n")
    _write_text(context.run_dir / "target_columns.txt", "\n".join(datamodule.target_columns) + "\n")
    _write_json(context.run_dir / "metrics.json", metrics_payload)
    importance.to_csv(context.run_dir / "feature_importance.csv", index=False)
    _write_text(context.run_dir / "experiment_report.md", _experiment_report(metrics_payload, importance))
    _save_model(context.run_dir / "model" / "model.pkl", model)
    _save_predictions(context.run_dir / "predictions" / "val_predictions.parquet", val_arrays, val_pred)
    _save_predictions(context.run_dir / "predictions" / "test_predictions.parquet", test_arrays, test_pred)

    return {
        "experiment_id": context.run_name,
        "curated_dir": curated_dir,
        "metrics": metrics_payload,
    }


def copy_curated_outputs(raw_dir: Path, curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    for file_name in CURATED_FILES:
        shutil.copy2(raw_dir / file_name, curated_dir / file_name)


def _split_sklearn_overrides(overrides: Sequence[str]) -> Tuple[List[str], List[str]]:
    sklearn_overrides = []
    hydra_overrides = []
    for override in overrides:
        if override.startswith("sklearn."):
            sklearn_overrides.append(override)
        else:
            hydra_overrides.append(override)
    return sklearn_overrides, hydra_overrides


def _require_task_type(model_name: str, task_type: str, allowed: set[str]) -> None:
    if task_type not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{model_name} does not support task_type={task_type}; expected one of {allowed_text}")


def _save_predictions(path: Path, arrays: DatasetArrays, y_pred: Iterable) -> None:
    frame = arrays.metadata.copy()
    frame["y_true"] = np.asarray(arrays.y)
    frame["y_pred"] = np.asarray(y_pred)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _save_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _experiment_report(metrics: Dict[str, Any], importance: pd.DataFrame) -> str:
    top_features = importance.head(10)
    lines = [
        f"# {metrics['experiment_id']}",
        "",
        "This is a real standalone non-MLP tabular baseline fit.",
        "",
        "## Run",
        "",
        f"- Dataset: `{metrics['dataset']}`",
        f"- Split: `{metrics['split']}`",
        f"- Task: `{metrics['task']}`",
        f"- Model: `{metrics['model']}`",
        f"- Fit status: `{metrics['fit_status']}`",
        f"- Feature count: {metrics['feature_count']}",
        f"- Target columns: `{', '.join(metrics['target_columns'])}`",
        f"- Train / Val / Test examples: {metrics['train_examples']} / {metrics['val_examples']} / {metrics['test_examples']}",
        "",
        "## Metrics",
        "",
        f"- Primary metric: `{metrics['primary_metric']}` ({metrics['metric_direction']})",
        f"- Val primary: {metrics['val_primary']:.6f}",
        f"- Test primary: {metrics['test_primary']:.6f}",
        "",
        "| Split | Metric | Value |",
        "|---|---|---:|",
    ]
    for split_name in ("val", "test"):
        for metric_name, value in metrics[f"{split_name}_metrics"].items():
            lines.append(f"| {split_name} | {metric_name} | {value:.6f} |")
    lines.extend([
        "",
        "## Top Feature Importance",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ])
    for row in top_features.itertuples(index=False):
        lines.append(f"| {row.rank} | `{row.feature}` | {row.importance:.6f} |")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- This model uses tabular manual features, not raw vibration sequences.",
        "- No hyperparameter sweep is performed in Step Y.",
        "- Raw model pickle and prediction parquet files are stored under artifacts and should not be committed.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()

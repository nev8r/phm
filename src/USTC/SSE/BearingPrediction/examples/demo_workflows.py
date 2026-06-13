"""
Notebook example workflow module

this file is for creating small runnable dataset fixtures and examples

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.api import (
    BaseTester,
    BaseTrainer,
    BearingEntity,
    BearingRulLabeler,
    BearingWindowDataset,
    CNN,
    CNNLSTMAttention,
    Evaluator,
    ExperimentConfig,
    ExperimentLoggerCallback,
    ExperimentTracker,
    FeatureSequenceRulLabeler,
    HuangRulScore,
    MAE,
    MLP,
    NormalizedRMSE,
    OverPredictionRate,
    PHM2012Loader,
    PHM2012Score,
    RMSE,
    SMAPE,
    WithinToleranceRate,
    XJTULoader,
)


def example_output_root() -> Path:
    """
    return writable example output directory

    Returns
    -------
    Path
        output root
    """

    output_root = Path(os.getenv("BEARING_EXAMPLE_OUTPUT_ROOT", "outputs/examples")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def demo_data_root() -> Path:
    """
    return demo data root

    Returns
    -------
    Path
        demo data root
    """

    data_root = example_output_root() / "demo_data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def create_demo_xjtu_dataset(base_dir: Path | None = None, *, sample_count: int = 6, signal_length: int = 128) -> Path:
    """
    create a tiny XJTU-SY-like directory that can be parsed by XJTULoader

    Parameters
    ----------
    base_dir : Path | None
        optional parent directory

    Returns
    -------
    Path
        dataset root
    """

    root = (base_dir or demo_data_root()) / "XJTU-SY_Bearing_Datasets"
    bearing_specs = [
        ("35Hz12kN", "Bearing1_1", 1.0),
        ("35Hz12kN", "Bearing1_2", 1.3),
        ("37.5Hz11kN", "Bearing2_1", 1.6),
    ]
    for condition_name, bearing_id, scale in bearing_specs:
        bearing_dir = root / condition_name / bearing_id
        bearing_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in range(1, sample_count + 1):
            signal_frame = _build_signal_frame(
                signal_length=signal_length,
                sample_id=sample_id,
                scale=scale,
                sample_count=sample_count,
                horizontal_column="Horizontal_vibration_signals",
                vertical_column="Vertical_vibration_signals",
            )
            signal_frame.to_csv(bearing_dir / f"{sample_id}.csv", index=False)
    return root


def create_demo_phm2012_dataset(base_dir: Path | None = None, *, sample_count: int = 6, signal_length: int = 128) -> Path:
    """
    create a tiny PHM2012/FEMTO-like directory that can be parsed by PHM2012Loader

    Parameters
    ----------
    base_dir : Path | None
        optional parent directory

    Returns
    -------
    Path
        dataset root
    """

    root = (base_dir or demo_data_root()) / "FEMTO"
    bearing_specs = [
        ("Bearing1_1", 9, 39, 10, 1.0),
        ("Bearing1_2", 9, 41, 0, 1.2),
        ("Bearing2_1", 10, 12, 20, 1.5),
    ]
    for bearing_id, hour_value, minute_value, second_value, scale in bearing_specs:
        bearing_dir = root / "Training_set" / "Learning_set" / bearing_id
        bearing_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in range(1, sample_count + 1):
            acceleration_frame = _build_phm_acceleration_frame(
                hour_value=hour_value,
                minute_value=minute_value,
                second_value=second_value + ((sample_id - 1) * 10),
                sample_id=sample_id,
                scale=scale,
                sample_count=sample_count,
                signal_length=signal_length,
            )
            acceleration_frame.to_csv(bearing_dir / f"acc_{sample_id:05d}.csv", index=False, header=False, sep=";")
            temperature_frame = _build_phm_temperature_frame(
                hour_value=hour_value,
                minute_value=minute_value,
                second_value=second_value + ((sample_id - 1) * 10),
                sample_id=sample_id,
            )
            temperature_frame.to_csv(bearing_dir / f"temp_{sample_id:05d}.csv", index=False, header=False, sep=";")
    return root


def run_generate_demo_datasets() -> dict[str, object]:
    """
    generate both tiny demo datasets

    Returns
    -------
    dict[str, object]
        output summary
    """

    xjtu_root = create_demo_xjtu_dataset()
    phm2012_root = create_demo_phm2012_dataset()
    return {
        "status": "OK",
        "xjtu_root": str(xjtu_root),
        "phm2012_root": str(phm2012_root),
    }


def run_xjtu_loader_overview() -> dict[str, object]:
    """
    run XJTU loader overview workflow

    Returns
    -------
    dict[str, object]
        output summary
    """

    data_root = create_demo_xjtu_dataset()
    loader = XJTULoader(data_root)
    entity = loader.load_entity("Bearing1_1")
    summary_frame = entity.samples[["sample_index", "timestamp", "elapsed_seconds", "rul", "source_file"]]
    output_path = example_output_root() / "xjtu_loader_overview" / "sample_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_path, index=False)
    return {
        "status": "OK",
        "entities": loader.list_entities(),
        "channels": entity.channel_names(),
        "metadata": entity.metadata,
        "summary_path": str(output_path),
    }


def run_phm2012_loader_overview() -> dict[str, object]:
    """
    run PHM2012 loader overview workflow

    Returns
    -------
    dict[str, object]
        output summary
    """

    data_root = create_demo_phm2012_dataset()
    loader = PHM2012Loader(data_root)
    entity = loader.load_entity("Bearing1_1")
    summary_frame = entity.samples[
        ["sample_index", "timestamp", "elapsed_seconds", "rul", "source_file", "temperature_file"]
    ]
    output_path = example_output_root() / "phm2012_loader_overview" / "sample_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output_path, index=False)
    return {
        "status": "OK",
        "entities": loader.list_entities(),
        "channels": entity.channel_names(),
        "metadata": entity.metadata,
        "summary_path": str(output_path),
    }


def run_xjtu_cnn_rul_training() -> dict[str, object]:
    """
    train a small CNN on demo XJTU-SY data

    Returns
    -------
    dict[str, object]
        output summary
    """

    output_root = example_output_root() / "xjtu_cnn_rul_training"
    entity = XJTULoader(create_demo_xjtu_dataset(sample_count=24, signal_length=256)).load_entity("Bearing1_1")
    window_size = 64
    dataset = BearingRulLabeler(window_size=window_size, stride=64, input_representation="raw_signal").label(
        entity,
        "Horizontal Vibration",
    )
    train_set, test_set = dataset.split_by_ratio(0.75)
    model = CNN(input_size=window_size, output_size=1, hidden_channels=4, dropout=0.1)
    trainer = _build_trainer(output_root, entity.dataset_name, "CNN", run_name="xjtu-cnn-demo")
    trainer.train(model, train_set, test_set)
    result = BaseTester(device="cpu", batch_size=4).test(model, test_set)
    metrics = Evaluator().add(MAE(), RMSE()).evaluate(result.targets, result.predictions)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "predictions.csv"
    metrics_path = output_root / "metrics.json"
    result.as_frame().to_csv(prediction_path, index=False)
    _write_json(metrics_path, metrics)
    return {"status": "OK", "metrics": metrics, "prediction_path": str(prediction_path)}


def run_phm2012_mlp_feature_training() -> dict[str, object]:
    """
    train a small MLP on demo PHM2012 feature windows

    Returns
    -------
    dict[str, object]
        output summary
    """

    output_root = example_output_root() / "phm2012_mlp_feature_training"
    entity = PHM2012Loader(create_demo_phm2012_dataset()).load_entity("Bearing1_1")
    dataset = BearingRulLabeler(window_size=64, stride=64, input_representation="features").label(
        entity,
        "Horizontal Vibration",
    )
    train_set, test_set = dataset.split_by_ratio(0.75)
    model = MLP(input_size=int(dataset.feature_frame.shape[1]), output_size=1, hidden_size=16, dropout=0.1)
    trainer = _build_trainer(output_root, entity.dataset_name, "MLP", run_name="phm2012-mlp-demo")
    trainer.train(model, train_set, test_set)
    result = BaseTester(device="cpu", batch_size=4).test(model, test_set)
    metrics = Evaluator().add(MAE(), RMSE()).evaluate(result.targets, result.predictions)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / "predictions.csv"
    metrics_path = output_root / "metrics.json"
    result.as_frame().to_csv(prediction_path, index=False)
    _write_json(metrics_path, metrics)
    return {"status": "OK", "metrics": metrics, "prediction_path": str(prediction_path)}


def run_cross_dataset_feature_export() -> dict[str, object]:
    """
    export feature tables from demo XJTU-SY and PHM2012 entities

    Returns
    -------
    dict[str, object]
        output summary
    """

    output_root = example_output_root() / "cross_dataset_feature_export"
    output_root.mkdir(parents=True, exist_ok=True)
    xjtu_entity = XJTULoader(create_demo_xjtu_dataset()).load_entity("Bearing1_1")
    phm_entity = PHM2012Loader(create_demo_phm2012_dataset()).load_entity("Bearing1_1")
    labeler = BearingRulLabeler(window_size=64, stride=64, input_representation="features")
    xjtu_dataset = labeler.label(xjtu_entity, "Horizontal Vibration")
    phm_dataset = labeler.label(phm_entity, "Horizontal Vibration")

    xjtu_features = xjtu_dataset.feature_frame.copy()
    xjtu_features["dataset_name"] = xjtu_entity.dataset_name
    xjtu_features["target_rul"] = xjtu_dataset.targets
    phm_features = phm_dataset.feature_frame.copy()
    phm_features["dataset_name"] = phm_entity.dataset_name
    phm_features["target_rul"] = phm_dataset.targets

    xjtu_path = output_root / "xjtu_features.csv"
    phm_path = output_root / "phm2012_features.csv"
    xjtu_features.to_csv(xjtu_path, index=False)
    phm_features.to_csv(phm_path, index=False)
    return {
        "status": "OK",
        "xjtu_rows": int(len(xjtu_features)),
        "phm2012_rows": int(len(phm_features)),
        "xjtu_path": str(xjtu_path),
        "phm2012_path": str(phm_path),
    }


def run_paper_cnn_lstm_attention_reproduction(
    *,
    xjtu_root: str | Path | None = None,
    phm2012_root: str | Path | None = None,
    max_samples_per_entity: int | None = None,
    prefer_real_data: bool = True,
) -> dict[str, object]:
    """
    run a CNN-LSTM-AM paper-style reproduction on XJTU-SY and PHM2012 data

    Returns
    -------
    dict[str, object]
        output summary
    """

    output_root = example_output_root() / "paper_cnn_lstm_attention"
    sample_limit = max_samples_per_entity or int(os.getenv("BEARING_EXAMPLE_MAX_SAMPLES", "36"))
    entities = _load_paper_reproduction_entities(
        xjtu_root=xjtu_root,
        phm2012_root=phm2012_root,
        max_samples_per_entity=sample_limit,
        prefer_real_data=prefer_real_data,
    )

    run_summaries: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    first_attention_run: dict[str, object] | None = None

    for entity, data_source in entities:
        dataset = _build_paper_feature_sequence_dataset(entity)
        train_set, test_set = dataset.split_by_ratio(0.75)
        for model_name, use_attention in [("CNN-LSTM-AM", True), ("CNN-LSTM", False)]:
            model = CNNLSTMAttention(
                feature_size=int(dataset.inputs.shape[-1]),
                output_size=1,
                cnn_channels=(8, 8, 8),
                lstm_hidden_size=12,
                lstm_layers=3,
                fc_hidden_sizes=(12, 6),
                dropout=0.1,
                use_attention=use_attention,
            )
            run_slug = f"{_slugify(entity.dataset_name)}-{_slugify(entity.entity_id)}-{_slugify(model_name)}"
            run_output_root = output_root / run_slug
            trainer = _build_trainer(
                run_output_root,
                entity.dataset_name,
                model_name,
                run_name=f"paper-{run_slug}",
            )
            training_result = trainer.train(model, train_set, test_set)
            test_result = BaseTester(device="cpu", batch_size=4).test(model, test_set)
            metrics = Evaluator().add(
                MAE(),
                RMSE(),
                NormalizedRMSE(),
                SMAPE(),
                HuangRulScore(),
                OverPredictionRate(),
                WithinToleranceRate(tolerance=0.10),
            ).evaluate(test_result.targets, test_result.predictions)
            metrics["phm2012_score_scaled"] = _scaled_phm2012_score(test_result.targets, test_result.predictions)

            run_output_root.mkdir(parents=True, exist_ok=True)
            prediction_path = run_output_root / "predictions.csv"
            metrics_path = run_output_root / "metrics.json"
            attention_path = run_output_root / "attention_weights.csv"
            test_result.as_frame().to_csv(prediction_path, index=False)
            _write_json(metrics_path, metrics)
            _write_attention_csv(attention_path, test_result.attention_weights)

            history_path = trainer.experiment_tracker.run_dir / "history.csv" if trainer.experiment_tracker is not None else run_output_root / "history.csv"
            run_summary = {
                "dataset_name": entity.dataset_name,
                "entity_id": entity.entity_id,
                "data_source": data_source,
                "model_name": model_name,
                "use_attention": use_attention,
                "metrics": metrics,
                "prediction_count": int(len(test_result.predictions)),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "attention_path": str(attention_path),
                "history_path": str(history_path),
                "feature_sequence_shape": list(dataset.inputs.shape),
                "epoch_count": int(len(training_result.history)),
            }
            run_summaries.append(run_summary)
            comparison_records.append(
                {
                    "dataset_name": entity.dataset_name,
                    "entity_id": entity.entity_id,
                    "data_source": data_source,
                    "model_name": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "normalized_rmse": metrics["normalized_rmse"],
                    "smape": metrics["smape"],
                    "huang_rul_score": metrics["huang_rul_score"],
                    "over_prediction_rate": metrics["over_prediction_rate"],
                    "within_10_percent_rate": metrics["within_10_percent_rate"],
                    "phm2012_score_scaled": metrics["phm2012_score_scaled"],
                    "prediction_count": int(len(test_result.predictions)),
                    "epoch_count": int(len(training_result.history)),
                    "history_path": str(history_path),
                }
            )
            if use_attention and first_attention_run is None:
                first_attention_run = run_summary

    comparison_path = output_root / "comparison_metrics.csv"
    comparison_frame = _add_attention_baseline_comparison_columns(pd.DataFrame.from_records(comparison_records))
    comparison_frame.to_csv(comparison_path, index=False)
    primary_run = first_attention_run or run_summaries[0]

    return {
        "status": "OK",
        "paper": "Life prediction method of rolling bearing based on CNN-LSTM-AM",
        "source": "https://www.extrica.com/article/23793",
        "used_dataset_count": len(entities),
        "trained_model_count": len(run_summaries),
        "runs": run_summaries,
        "comparison_path": str(comparison_path),
        "metrics": primary_run["metrics"],
        "prediction_path": primary_run["prediction_path"],
        "attention_path": primary_run["attention_path"],
        "feature_sequence_shape": primary_run["feature_sequence_shape"],
    }


def _build_trainer(output_root: Path, dataset_name: str, model_name: str, *, run_name: str) -> BaseTrainer:
    max_epochs = int(os.getenv("BEARING_EXAMPLE_EPOCHS", "2"))
    tracker = ExperimentTracker(
        output_root / "experiments",
        ExperimentConfig(
            run_name=run_name,
            dataset_name=dataset_name,
            model_name=model_name,
            optimizer_name="Adam",
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=max_epochs,
            batch_size=4,
            sampling_strategy="chronological",
            prediction_mode="direct",
        ),
    )
    return BaseTrainer(
        device="cpu",
        callbacks=[ExperimentLoggerCallback()],
        experiment_tracker=tracker,
        max_epochs=max_epochs,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        shuffle_train=False,
    )


def _load_paper_reproduction_entities(
    *,
    xjtu_root: str | Path | None,
    phm2012_root: str | Path | None,
    max_samples_per_entity: int,
    prefer_real_data: bool,
) -> list[tuple[BearingEntity, str]]:
    xjtu_data_root = _resolve_xjtu_root(xjtu_root, prefer_real_data)
    phm_data_root = _resolve_phm2012_root(phm2012_root, prefer_real_data)
    data_source = "real_or_provided_files" if prefer_real_data and xjtu_data_root.exists() and phm_data_root.exists() else "generated_demo_files"

    xjtu_loader = XJTULoader(xjtu_data_root)
    phm_loader = PHM2012Loader(phm_data_root)
    xjtu_entity = xjtu_loader.load_entity(_select_entity_id(xjtu_loader, ["Bearing1_5", "Bearing2_4", "Bearing1_1", "Bearing2_1"]))
    phm_entity = phm_loader.load_entity(_select_entity_id(phm_loader, ["Bearing3_1", "Bearing1_2", "Bearing2_1", "Bearing1_1"]))
    return [
        (_sample_entity_snapshots(xjtu_entity, max_samples_per_entity), data_source),
        (_sample_entity_snapshots(phm_entity, max_samples_per_entity), data_source),
    ]


def _build_paper_feature_sequence_dataset(entity: BearingEntity) -> BearingWindowDataset:
    channel_name = "Horizontal Vibration"
    signal_lengths = [len(signal_values) for signal_values in entity.get_channel(channel_name)]
    window_size = min(1024, min(signal_lengths))
    if window_size < 16:
        raise ValueError(f"{entity.entity_id} has snapshots shorter than 16 points")
    return FeatureSequenceRulLabeler(sequence_length=5, window_size=window_size, stride=window_size).label(
        entity,
        channel_name,
    )


def _resolve_xjtu_root(root: str | Path | None, prefer_real_data: bool) -> Path:
    if root is not None:
        return Path(root)
    candidate_root = Path("data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets")
    if prefer_real_data and candidate_root.exists():
        return candidate_root
    return create_demo_xjtu_dataset(sample_count=24, signal_length=256)


def _resolve_phm2012_root(root: str | Path | None, prefer_real_data: bool) -> Path:
    if root is not None:
        return Path(root)
    candidate_root = Path("data/external/phm2012/final")
    if prefer_real_data and candidate_root.exists():
        return candidate_root
    return create_demo_phm2012_dataset(sample_count=24, signal_length=256)


def _select_entity_id(loader: XJTULoader | PHM2012Loader, preferred_ids: list[str]) -> str:
    available_entities = loader.list_entities()
    for entity_id in preferred_ids:
        if entity_id in available_entities:
            return entity_id
    if not available_entities:
        raise ValueError(f"no bearing entities found under {loader.data_root}")
    return available_entities[0]


def _sample_entity_snapshots(entity: BearingEntity, max_samples: int) -> BearingEntity:
    if max_samples <= 0 or len(entity.samples) <= max_samples:
        return entity
    sample_indices = np.linspace(0, len(entity.samples) - 1, max_samples, dtype=int)
    sampled_frame = entity.samples.iloc[np.unique(sample_indices)].reset_index(drop=True).copy()
    sampled_metadata = {**entity.metadata, "source_sample_count": int(len(entity.samples)), "used_sample_count": int(len(sampled_frame))}
    return BearingEntity(
        entity_id=entity.entity_id,
        dataset_name=entity.dataset_name,
        samples=sampled_frame,
        sample_rate=entity.sample_rate,
        metadata=sampled_metadata,
    )


def _scaled_phm2012_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    target_scale = max(float(np.max(targets) - np.min(targets)), 1.0)
    scaled_targets = (targets - np.min(targets)) / target_scale * 100.0
    scaled_predictions = (predictions - np.min(targets)) / target_scale * 100.0
    return PHM2012Score()(scaled_targets, scaled_predictions)


def _add_attention_baseline_comparison_columns(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    comparison_frame = comparison_frame.copy()
    comparison_frame["rmse_reduction_pct"] = np.nan
    comparison_frame["huang_score_change_pct"] = np.nan
    for dataset_name, dataset_frame in comparison_frame.groupby("dataset_name"):
        model_rows = dataset_frame.set_index("model_name")
        if "CNN-LSTM-AM" not in model_rows.index or "CNN-LSTM" not in model_rows.index:
            continue
        attention_row = model_rows.loc["CNN-LSTM-AM"]
        baseline_row = model_rows.loc["CNN-LSTM"]
        rmse_reduction = _safe_percent_change(baseline_row["rmse"] - attention_row["rmse"], baseline_row["rmse"])
        score_change = _safe_percent_change(attention_row["huang_rul_score"] - baseline_row["huang_rul_score"], baseline_row["huang_rul_score"])
        dataset_mask = comparison_frame["dataset_name"] == dataset_name
        comparison_frame.loc[dataset_mask, "rmse_reduction_pct"] = rmse_reduction
        comparison_frame.loc[dataset_mask, "huang_score_change_pct"] = score_change
    return comparison_frame


def _safe_percent_change(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-8:
        return 0.0
    return float((numerator / denominator) * 100.0)


def _write_attention_csv(output_path: Path, attention_weights: np.ndarray | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if attention_weights is None:
        pd.DataFrame(columns=["attention_disabled"]).to_csv(output_path, index=False)
        return
    column_names = [f"attention_t{index}" for index in range(attention_weights.shape[1])]
    pd.DataFrame(attention_weights, columns=column_names).to_csv(output_path, index=False)


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "-").replace("_", "-")


def _write_json(output_path: Path, values: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_signal_frame(
    *,
    signal_length: int,
    sample_id: int,
    scale: float,
    sample_count: int,
    horizontal_column: str,
    vertical_column: str,
) -> pd.DataFrame:
    time_axis = np.linspace(0.0, 1.0, signal_length, endpoint=False)
    degradation = sample_id / max(sample_count, 1)
    horizontal_signal = (
        scale * (0.6 + degradation) * np.sin(2.0 * np.pi * (8.0 + sample_id) * time_axis)
        + 0.08 * sample_id * np.sin(2.0 * np.pi * 28.0 * time_axis)
    )
    vertical_signal = (
        scale * (0.4 + degradation) * np.cos(2.0 * np.pi * (5.0 + sample_id) * time_axis)
        + 0.05 * sample_id * np.cos(2.0 * np.pi * 20.0 * time_axis)
    )
    return pd.DataFrame({horizontal_column: horizontal_signal.astype(float), vertical_column: vertical_signal.astype(float)})


def _build_phm_acceleration_frame(
    *,
    hour_value: int,
    minute_value: int,
    second_value: int,
    sample_id: int,
    scale: float,
    sample_count: int,
    signal_length: int,
) -> pd.DataFrame:
    signal_frame = _build_signal_frame(
        signal_length=signal_length,
        sample_id=sample_id,
        scale=scale,
        sample_count=sample_count,
        horizontal_column="horizontal",
        vertical_column="vertical",
    )
    microseconds = np.arange(0, signal_frame.shape[0] * 39, 39)
    return pd.DataFrame(
        {
            0: np.full(signal_frame.shape[0], hour_value),
            1: np.full(signal_frame.shape[0], minute_value),
            2: np.full(signal_frame.shape[0], second_value),
            3: microseconds,
            4: signal_frame["horizontal"].to_numpy(dtype=float),
            5: signal_frame["vertical"].to_numpy(dtype=float),
        }
    )


def _build_phm_temperature_frame(
    *,
    hour_value: int,
    minute_value: int,
    second_value: int,
    sample_id: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            0: np.full(8, hour_value),
            1: np.full(8, minute_value),
            2: np.full(8, second_value),
            3: np.arange(8),
            4: np.linspace(38.0 + sample_id, 39.2 + sample_id, 8),
        }
    )

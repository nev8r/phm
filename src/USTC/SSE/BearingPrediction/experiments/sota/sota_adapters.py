"""
Open-source SOTA adapter module

this file is for describing external SOTA reproduction adapters

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from USTC.SSE.BearingPrediction.experiments.sota.sota_protocol import calculate_gap_percent


@dataclass(frozen=True)
class ExternalSotaAdapter:
    """
    Description of an external open-source SOTA reproduction route.

    Parameters
    ----------
    name : str
        adapter name
    repository_url : str
        source repository url
    source_commit : str
        pinned commit
    run_command : str
        external command
    environment_status : str
        local environment status
    notes : str
        caveats
    """

    name: str
    repository_url: str
    source_commit: str
    run_command: str
    environment_status: str
    notes: str

    def can_run_in_project_environment(self) -> bool:
        """
        return whether this adapter can run inside the current project environment

        Returns
        -------
        bool
            True when runnable locally
        """

        return self.environment_status == "runnable"


def default_external_adapters() -> list[ExternalSotaAdapter]:
    """
    list pinned external SOTA reproduction routes.

    Returns
    -------
    list[ExternalSotaAdapter]
        adapter descriptions
    """

    return [
        ExternalSotaAdapter(
            name="AutoRUL",
            repository_url="https://github.com/Ennosigaeon/auto-sktime",
            source_commit="fe277d21104be8d2e4bd34db7ed995547007e55b",
            run_command="git checkout tags/v0.1.0 -b autorul && cd scripts && python remaining_useful_lifetime.py femto_bearing",
            environment_status="external_dependency_stack",
            notes="MIT licensed AutoRUL implementation. Runtime can be long because it performs AutoML search; target is locked for PRONOSTIA/FEMTO.",
        ),
        ExternalSotaAdapter(
            name="RULSurv RSF",
            repository_url="https://github.com/thecml/rulsurv",
            source_commit="6365e0832de9724a5bcbbac4557c6643dfb78d91",
            run_command="python src/make_dataset.py && python src/run_cross_validation.py && python src/predict_isd_curves.py",
            environment_status="external_dependency_stack",
            notes="Requires Python 3.9 era TensorFlow/scikit-survival/pycox stack; target is locked but not claimed as rerun in this uv Python 3.11 project.",
        ),
        ExternalSotaAdapter(
            name="GNN RUL Benchmarking",
            repository_url="https://github.com/Frank-Wang-oss/GNN_RUL_Benchmarking",
            source_commit="9325667ed34976452e9323728e33a29fe0f98b5e",
            run_command="python main.py --experiment_description exp1 --run_description run_1 --GNN_method FC_STGNN --dataset PHM2012 --num_runs 5",
            environment_status="external_dependency_stack",
            notes="Requires repo-specific preprocessed data and PyTorch/skorch versions; target is locked as a follow-up external run.",
        ),
        ExternalSotaAdapter(
            name="Weibull KIML",
            repository_url="https://github.com/tvhahn/weibull-knowledge-informed-ml",
            source_commit="c430d4b710450a1661e528675a6c1ccc64bc98e2",
            run_command="make train_femto && make summarize_femto_models && make figures_results",
            environment_status="external_dependency_stack",
            notes="MIT licensed physics/reliability-prior reference for PRONOSTIA/FEMTO; random search cost is high.",
        ),
    ]


@dataclass(frozen=True)
class RulSurvRsfPortConfig:
    """
    Configuration for the RULSurv Random Survival Forest port.

    Parameters
    ----------
    xjtu_root : Path
        XJTU-SY dataset root
    output_dir : Path
        evidence output directory
    """

    xjtu_root: Path
    output_dir: Path
    condition_dir: str = "35Hz12kN"
    target_id: str = "rulsurv-xjtu-high-rsf-true-mae"
    source_url: str = "https://github.com/thecml/rulsurv"
    source_commit: str = "6365e0832de9724a5bcbbac4557c6643dfb78d91"
    target_true_mae_minutes: float = 12.6
    censoring_level: float = 0.25
    n_splits: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    n_estimators: int = 200
    min_samples_leaf: int = 20
    max_depth: int = 7
    train_bearings: tuple[str, ...] = ("Bearing1_1", "Bearing1_2", "Bearing1_4", "Bearing1_5")
    test_bearing: str = "Bearing1_3"


class RulSurvRsfPortAdapter:
    """
    Run a local port of the RULSurv RSF protocol on XJTU-SY condition 1.
    """

    frequency_bands = {
        "35Hz12kN": ([12, 34, 71, 107, 171], [14, 36, 73, 109, 173]),
        "37.5Hz11kN": ([13, 36, 76, 114, 183], [15, 38, 78, 116, 185]),
        "40Hz10kN": ([14, 39, 82, 122, 195], [15, 41, 84, 124, 197]),
    }
    feature_names = (
        "mean",
        "std",
        "skew",
        "kurtosis",
        "entropy",
        "rms",
        "max",
        "p2p",
        "crest",
        "clearance",
        "shape",
        "impulse",
        "FoH",
        "FiH",
        "FrH",
        "FrpH",
        "FcaH",
        "Fo",
        "Fi",
        "Fr",
        "Frp",
        "Fca",
        "noise",
    )

    def __init__(self, config: RulSurvRsfPortConfig) -> None:
        self.config = config

    def run(self) -> dict[str, str]:
        """
        run original-protocol CV and project holdout migration.

        Returns
        -------
        dict[str, str]
            output artifact paths
        """

        try:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv
        except ImportError as exc:
            raise RuntimeError(
                "RULSurv RSF port requires scikit-survival. "
                "Run with: uv run --with scikit-survival python scripts/run_rulsurv_rsf_port.py"
            ) from exc

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._build_condition_frame()
        audit_frame = self._build_snapshot_audit_frame(dataset)
        original_metrics, original_predictions = self._run_original_cv(dataset, RandomSurvivalForest, Surv)
        holdout_metrics, holdout_predictions = self._run_project_holdout(dataset, RandomSurvivalForest, Surv)

        metrics_frame = pd.concat([original_metrics, holdout_metrics], ignore_index=True)
        predictions_frame = pd.concat([original_predictions, holdout_predictions], ignore_index=True)
        summary_frame = self._build_summary_frame(metrics_frame)

        metrics_path = self.config.output_dir / "rulsurv_rsf_port_metrics.csv"
        predictions_path = self.config.output_dir / "rulsurv_rsf_port_predictions.csv"
        summary_path = self.config.output_dir / "rulsurv_rsf_port_summary.csv"
        audit_path = self.config.output_dir / "rulsurv_rsf_port_snapshot_audit.csv"
        config_path = self.config.output_dir / "rulsurv_rsf_port_config.json"
        metrics_frame.to_csv(metrics_path, index=False)
        predictions_frame.to_csv(predictions_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        audit_frame.to_csv(audit_path, index=False)
        config_path.write_text(json.dumps(self._json_ready_config(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "metrics_path": self._display_path(metrics_path),
            "predictions_path": self._display_path(predictions_path),
            "summary_path": self._display_path(summary_path),
            "audit_path": self._display_path(audit_path),
            "config_path": self._display_path(config_path),
        }

    def _run_original_cv(self, dataset: pd.DataFrame, model_class, surv_builder) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_frame, feature_columns = self._feature_matrix(dataset)
        metric_records: list[dict[str, object]] = []
        prediction_records: list[dict[str, object]] = []
        for seed in self.config.seeds:
            censored = self._add_random_censoring(dataset, seed=seed)
            kfold = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=seed)
            for fold_index, (train_index, test_index) in enumerate(kfold.split(feature_frame)):
                model, predictions = self._fit_predict_survival(
                    model_class=model_class,
                    surv_builder=surv_builder,
                    features=feature_frame,
                    feature_columns=feature_columns,
                    event_values=censored["Event"].to_numpy(dtype=bool),
                    survival_times=censored["Survival_time"].to_numpy(dtype=float),
                    true_times=censored["TrueTime"].to_numpy(dtype=float),
                    train_index=train_index,
                    test_index=test_index,
                    seed=seed,
                )
                del model
                metric_records.append(
                    self._metric_record(
                        protocol="rulsurv_original_25pct_censored_cv",
                        seed=seed,
                        fold_index=fold_index,
                        split_name="five_fold_row_level_cv",
                        predictions=predictions,
                    )
                )
                prediction_records.extend(
                    self._prediction_records(
                        protocol="rulsurv_original_25pct_censored_cv",
                        seed=seed,
                        fold_index=fold_index,
                        split_name="five_fold_row_level_cv",
                        predictions=predictions,
                    )
                )
        return pd.DataFrame.from_records(metric_records), pd.DataFrame.from_records(prediction_records)

    def _run_project_holdout(self, dataset: pd.DataFrame, model_class, surv_builder) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_frame, feature_columns = self._feature_matrix(dataset)
        train_index = dataset.index[dataset["bearing_id"].isin(self.config.train_bearings)].to_numpy()
        test_index = dataset.index[dataset["bearing_id"] == self.config.test_bearing].to_numpy()
        metric_records: list[dict[str, object]] = []
        prediction_records: list[dict[str, object]] = []
        for seed in self.config.seeds:
            model, predictions = self._fit_predict_survival(
                model_class=model_class,
                surv_builder=surv_builder,
                features=feature_frame,
                feature_columns=feature_columns,
                event_values=dataset["Event"].to_numpy(dtype=bool),
                survival_times=dataset["Survival_time"].to_numpy(dtype=float),
                true_times=dataset["TrueTime"].to_numpy(dtype=float),
                train_index=train_index,
                test_index=test_index,
                seed=seed,
            )
            del model
            metric_records.append(
                self._metric_record(
                    protocol="project_bearing1_3_holdout_migration",
                    seed=seed,
                    fold_index=0,
                    split_name="train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3",
                    predictions=predictions,
                )
            )
            prediction_records.extend(
                self._prediction_records(
                    protocol="project_bearing1_3_holdout_migration",
                    seed=seed,
                    fold_index=0,
                    split_name="train_Bearing1_1_1_2_1_4_1_5_test_Bearing1_3",
                    predictions=predictions,
                )
            )
        return pd.DataFrame.from_records(metric_records), pd.DataFrame.from_records(prediction_records)

    def _fit_predict_survival(
        self,
        *,
        model_class,
        surv_builder,
        features: pd.DataFrame,
        feature_columns: list[str],
        event_values: np.ndarray,
        survival_times: np.ndarray,
        true_times: np.ndarray,
        train_index: np.ndarray,
        test_index: np.ndarray,
        seed: int,
    ) -> tuple[object, pd.DataFrame]:
        scaler = StandardScaler()
        train_features = pd.DataFrame(scaler.fit_transform(features.iloc[train_index]), columns=feature_columns)
        test_features = pd.DataFrame(scaler.transform(features.iloc[test_index]), columns=feature_columns)
        train_targets = surv_builder.from_arrays(event=event_values[train_index], time=survival_times[train_index])
        model = model_class(
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_samples_leaf,
            min_samples_split=max(2, self.config.min_samples_leaf * 2),
            max_depth=self.config.max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_features, train_targets)
        predicted_times = self._predict_median_survival_time(model.predict_survival_function(test_features))
        return model, pd.DataFrame(
            {
                "row_index": test_index,
                "bearing_id": self._row_values("bearing_id", test_index),
                "true_time_minutes": true_times[test_index],
                "observed_survival_time_minutes": survival_times[test_index],
                "event": event_values[test_index],
                "predicted_time_minutes": predicted_times,
            }
        )

    def _build_condition_frame(self) -> pd.DataFrame:
        frames = [self._bearing_feature_frame(bearing_id) for bearing_id in ("Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5")]
        condition_frame = pd.concat(frames, ignore_index=True)
        # Exclude only the failure instant (TTE = 0). Positive one-minute RUL
        # snapshots remain valid survival samples and should not be dropped.
        condition_frame = condition_frame[condition_frame["TrueTime"] > 0.0].reset_index(drop=True)
        self._condition_frame = condition_frame
        return condition_frame

    def _bearing_feature_frame(self, bearing_id: str) -> pd.DataFrame:
        bearing_dir = self.config.xjtu_root / self.config.condition_dir / bearing_id
        signal_paths = sorted(bearing_dir.glob("*.csv"), key=lambda path: int(re.sub(r"\D", "", path.stem) or 0))
        if not signal_paths:
            raise ValueError(f"no XJTU csv files found under {bearing_dir}")
        records = []
        for sample_number, signal_path in enumerate(signal_paths):
            signal_frame = pd.read_csv(signal_path)
            signal_values = signal_frame.iloc[:, :2].to_numpy(dtype=float)
            records.append(self._extract_rulsurv_features(signal_values))
        feature_frame = pd.DataFrame.from_records(records)
        lifetime = len(feature_frame) - 1
        feature_frame["TrueTime"] = np.arange(lifetime, -1, -1, dtype=float)
        feature_frame["Survival_time"] = feature_frame["TrueTime"]
        feature_frame["Event"] = True
        feature_frame["elapsed_min"] = np.arange(len(feature_frame), dtype=float)
        feature_frame["bearing_id"] = bearing_id
        return feature_frame

    def _extract_rulsurv_features(self, signal_values: np.ndarray) -> dict[str, float]:
        band_start, band_stop = self.frequency_bands[self.config.condition_dir]
        values = np.asarray(signal_values, dtype=float)
        abs_values = np.abs(values)
        rms = np.sqrt(np.mean(values**2, axis=0))
        mean_abs = np.mean(abs_values, axis=0)
        max_abs = np.max(abs_values, axis=0)
        p2p = np.max(values, axis=0) - np.min(values, axis=0)
        clearance = np.mean(np.sqrt(abs_values), axis=0) ** 2
        hilbert_envelope = np.abs(hilbert(values, axis=0))
        fft_hilbert = self._one_sided_fft(hilbert_envelope)
        fft_abs = self._one_sided_fft(abs_values)
        record: dict[str, float] = {}
        for channel_index, prefix in enumerate(["B1", "B2"]):
            channel_values = values[:, channel_index]
            channel_abs = abs_values[:, channel_index]
            channel_mean_abs = mean_abs[channel_index]
            channel_rms = rms[channel_index]
            base_values = {
                "mean": channel_mean_abs,
                "std": float(np.std(channel_values, ddof=1)),
                "skew": float(pd.Series(channel_values).skew()),
                "kurtosis": float(pd.Series(channel_values).kurtosis()),
                "entropy": self._signal_entropy(channel_values),
                "rms": channel_rms,
                "max": max_abs[channel_index],
                "p2p": p2p[channel_index],
                "crest": max_abs[channel_index] / max(channel_rms, 1e-12),
                "clearance": clearance[channel_index],
                "shape": channel_rms / max(channel_mean_abs, 1e-12),
                "impulse": max_abs[channel_index] / max(channel_mean_abs, 1e-12),
                "noise": float(np.mean(fft_abs[:, channel_index])),
            }
            for name, value in base_values.items():
                record[f"{prefix}_{name}"] = float(value)
            for band_index, band_name in enumerate(["FoH", "FiH", "FrH", "FrpH", "FcaH"]):
                record[f"{prefix}_{band_name}"] = float(np.mean(fft_hilbert[band_start[band_index] : band_stop[band_index], channel_index]))
            for band_index, band_name in enumerate(["Fo", "Fi", "Fr", "Frp", "Fca"]):
                record[f"{prefix}_{band_name}"] = float(np.mean(fft_abs[band_start[band_index] : band_stop[band_index], channel_index]))
        return record

    def _feature_matrix(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        feature_columns = [
            column_name
            for column_name in dataset.columns
            if (column_name.startswith("B1_") or column_name.startswith("B2_")) and not column_name.endswith("_Event") and not column_name.endswith("_Survival_time")
        ]
        feature_frame = dataset[feature_columns].replace([np.inf, -np.inf], np.nan)
        valid_columns = feature_frame.columns[feature_frame.notna().all()]
        feature_frame = feature_frame[valid_columns]
        variable_columns = feature_frame.columns[feature_frame.std() > 1e-12]
        feature_frame = feature_frame[variable_columns].copy()
        feature_frame["elapsed_min"] = dataset["elapsed_min"].to_numpy(dtype=float)
        feature_frame["log_elapsed_min"] = np.log1p(dataset["elapsed_min"].to_numpy(dtype=float))
        feature_frame["sqrt_elapsed_min"] = np.sqrt(dataset["elapsed_min"].to_numpy(dtype=float))
        return feature_frame, list(feature_frame.columns)

    def _add_random_censoring(self, dataset: pd.DataFrame, *, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        censored = dataset.copy()
        samples_to_censor = censored.sample(frac=self.config.censoring_level, random_state=0).index
        for row_index in samples_to_censor:
            survival_time = int(max(1, censored.loc[row_index, "Survival_time"]))
            censored_time = int(rng.integers(0, survival_time)) if survival_time > 1 else 1
            censored.loc[row_index, "Survival_time"] = max(1, censored_time)
            censored.loc[row_index, "Event"] = False
        return censored

    def _metric_record(
        self,
        *,
        protocol: str,
        seed: int,
        fold_index: int,
        split_name: str,
        predictions: pd.DataFrame,
    ) -> dict[str, object]:
        true_values = predictions["true_time_minutes"].to_numpy(dtype=float)
        predicted_values = predictions["predicted_time_minutes"].to_numpy(dtype=float)
        return {
            "protocol": protocol,
            "seed": seed,
            "fold_index": fold_index,
            "split_name": split_name,
            "mae_true_minutes": float(mean_absolute_error(true_values, predicted_values)),
            "rmse_true_minutes": float(mean_squared_error(true_values, predicted_values) ** 0.5),
            "r2_true": float(r2_score(true_values, predicted_values)),
            "prediction_count": int(len(predictions)),
        }

    def _prediction_records(
        self,
        *,
        protocol: str,
        seed: int,
        fold_index: int,
        split_name: str,
        predictions: pd.DataFrame,
    ) -> list[dict[str, object]]:
        records = predictions.copy()
        records.insert(0, "split_name", split_name)
        records.insert(0, "fold_index", fold_index)
        records.insert(0, "seed", seed)
        records.insert(0, "protocol", protocol)
        return records.to_dict("records")

    def _build_summary_frame(self, metrics: pd.DataFrame) -> pd.DataFrame:
        summary_records = []
        for protocol, protocol_frame in metrics.groupby("protocol", sort=False):
            seed_means = protocol_frame.groupby("seed")["mae_true_minutes"].mean()
            prediction_count = int(protocol_frame.groupby("seed")["prediction_count"].sum().median())
            local_value = float(seed_means.min())
            local_mean = float(seed_means.mean())
            local_std = float(seed_means.std(ddof=0))
            split_description = (
                "RULSurv-compatible XJTU-SY condition 1 row-level 5-fold CV with 25% random censoring; "
                "rows from the same bearing can appear in different folds, so this is not a held-out-bearing generalization split"
                if protocol == "rulsurv_original_25pct_censored_cv"
                else "Project migration split: train Bearing1_1/Bearing1_2/Bearing1_4/Bearing1_5, test Bearing1_3"
            )
            status = "PROTOCOL_PASS" if protocol == "rulsurv_original_25pct_censored_cv" and calculate_gap_percent(
                local_value=local_mean,
                target_value=self.config.target_true_mae_minutes,
                higher_is_better=False,
            ) <= 25.0 else "NEEDS_OPTIMIZATION"
            summary_records.append(
                {
                    "target_id": self.config.target_id,
                    "experiment_name": f"RULSurv-RSF-port-{protocol}",
                    "method_name": "RULSurv RSF",
                    "local_method_name": "RULSurv RSF port",
                    "dataset_name": "XJTU-SY",
                    "condition_name": "highest_load_censored_cv" if protocol == "rulsurv_original_25pct_censored_cv" else "condition_1_35Hz12kN",
                    "metric_name": "true_mae_minutes",
                    "target_value": self.config.target_true_mae_minutes,
                    "local_value": local_value,
                    "local_mean": local_mean,
                    "local_std": local_std,
                    "gap_percent": calculate_gap_percent(local_value=local_value, target_value=self.config.target_true_mae_minutes, higher_is_better=False),
                    "mean_gap_percent": calculate_gap_percent(local_value=local_mean, target_value=self.config.target_true_mae_minutes, higher_is_better=False),
                    "metric_direction": "lower",
                    "run_count": int(seed_means.size),
                    "seeds": ",".join(str(seed) for seed in seed_means.index),
                    "prediction_count": prediction_count,
                    "source_url": self.config.source_url,
                    "source_commit": self.config.source_commit,
                    "split_description": split_description,
                    "evidence_path": self._display_path(self.config.output_dir / "rulsurv_rsf_port_metrics.csv"),
                    "status": status,
                    "notes": (
                        "Local Python 3.11 port of the RULSurv RSF route using RULSurv-style time/frequency features. "
                        "Original repo dependencies are not vendored; scikit-survival is supplied at run time with uv --with. "
                        "Interpret the original-protocol CV separately from the project holdout migration result."
                    ),
                }
            )
        return pd.DataFrame.from_records(summary_records)

    def _row_values(self, column_name: str, row_index: np.ndarray) -> np.ndarray:
        return self._condition_frame.loc[row_index, column_name].to_numpy()

    def _build_snapshot_audit_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        records = []
        for bearing_id in ("Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"):
            bearing_dir = self.config.xjtu_root / self.config.condition_dir / bearing_id
            raw_snapshot_count = len(list(bearing_dir.glob("*.csv")))
            used_snapshot_count = int((dataset["bearing_id"] == bearing_id).sum())
            records.append(
                {
                    "dataset_name": "XJTU-SY",
                    "condition_name": self.config.condition_dir,
                    "bearing_id": bearing_id,
                    "raw_snapshot_count": raw_snapshot_count,
                    "used_snapshot_count": used_snapshot_count,
                    "excluded_snapshot_count": raw_snapshot_count - used_snapshot_count,
                    "exclusion_reason": "Only TTE=0 failure instant is excluded; all positive-RUL snapshots are used.",
                    "uses_sampling_cap": False,
                }
            )
        total_raw = sum(int(record["raw_snapshot_count"]) for record in records)
        total_used = sum(int(record["used_snapshot_count"]) for record in records)
        records.append(
            {
                "dataset_name": "XJTU-SY",
                "condition_name": self.config.condition_dir,
                "bearing_id": "TOTAL",
                "raw_snapshot_count": total_raw,
                "used_snapshot_count": total_used,
                "excluded_snapshot_count": total_raw - total_used,
                "exclusion_reason": "One TTE=0 failure instant per bearing is excluded.",
                "uses_sampling_cap": False,
            }
        )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _one_sided_fft(values: np.ndarray) -> np.ndarray:
        fft_values = np.abs(np.fft.fft(values, axis=0) / len(values))
        fft_values = 2.0 * fft_values[: int(len(values) / 2 + 1), :]
        fft_values[0, :] = fft_values[0, :] / 2.0
        return fft_values

    @staticmethod
    def _signal_entropy(signal_values: np.ndarray) -> float:
        counts, _ = np.histogram(signal_values, bins=500)
        return float(entropy(counts + 1e-12))

    @staticmethod
    def _predict_median_survival_time(survival_functions) -> np.ndarray:
        predictions = []
        for survival_function in survival_functions:
            times = survival_function.x.astype(float)
            probabilities = survival_function.y.astype(float)
            below_median = np.where(probabilities <= 0.5)[0]
            if len(below_median):
                predictions.append(float(times[below_median[0]]))
            else:
                predictions.append(float(np.trapezoid(probabilities, times)))
        return np.asarray(predictions, dtype=float)

    def _json_ready_config(self) -> dict[str, object]:
        config_dict = asdict(self.config)
        for key in ["xjtu_root", "output_dir"]:
            config_dict[key] = self._display_path(Path(config_dict[key]))
        return config_dict

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

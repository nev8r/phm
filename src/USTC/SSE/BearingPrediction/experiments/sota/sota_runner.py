"""
Open-source SOTA evidence builder

this file is for building SOTA target and reproduction summary artifacts

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from USTC.SSE.BearingPrediction.experiments.sota.sota_protocol import (
    REPRODUCTION_COLUMNS,
    TARGET_COLUMNS,
    SotaReproductionRecord,
    SotaTargetRecord,
    validate_reproduction_frame,
    validate_target_frame,
)


class SotaEvidenceBuilder:
    """
    Build open-source SOTA target and local reproduction evidence tables.
    """

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.evidence_dir = self.project_root / "docs" / "reproduction-evidence"

    def default_targets(self) -> pd.DataFrame:
        """
        build default SOTA target table.

        Returns
        -------
        pd.DataFrame
            target records
        """

        targets = [
            SotaTargetRecord(
                target_id="autorul-pronostia-femto-rmse",
                method_name="AutoRUL",
                dataset_name="PRONOSTIA",
                condition_name="femto_bearing",
                metric_name="rmse",
                target_value=22.52,
                metric_direction="lower",
                source_type="open_source_repo_with_paper",
                source_url="https://github.com/Ennosigaeon/auto-sktime",
                source_commit="fe277d21104be8d2e4bd34db7ed995547007e55b",
                split_description=(
                    "AutoRUL femto_bearing benchmark from auto-sktime tag v0.1.0; "
                    "remaining_useful_lifetime.py combines train and validation folds then evaluates the benchmark test fold."
                ),
                reproducibility_status="open_source_external_env_required",
                license_name="MIT",
                run_command="git checkout tags/v0.1.0 -b autorul && cd scripts && python remaining_useful_lifetime.py femto_bearing",
                notes="AutoRUL paper reports PRONOSTIA test RMSE 22.52 +/- 5.68 over ten repetitions; strong target for tsfresh/sklearn/sktime style baseline work.",
            ),
            SotaTargetRecord(
                target_id="rulsurv-xjtu-high-rsf-true-mae",
                method_name="RULSurv RSF",
                dataset_name="XJTU-SY",
                condition_name="highest_load_censored_cv",
                metric_name="true_mae_minutes",
                target_value=12.6,
                metric_direction="lower",
                source_type="open_source_repo_with_paper",
                source_url="https://github.com/thecml/rulsurv",
                source_commit="6365e0832de9724a5bcbbac4557c6643dfb78d91",
                split_description="RULSurv five-fold censored survival protocol on XJTU-SY load strata",
                reproducibility_status="open_source_port_reproduced",
                license_name="MIT",
                run_command="uv run --with scikit-survival python scripts/run_rulsurv_rsf_port.py",
                notes=(
                    "RULSurv reports RSF true MAE 12.6 +/- 0.8 minutes for high load at 25% censoring. "
                    "This project records a Python 3.11 port with RULSurv-style features and scikit-survival RSF; "
                    "original repo dependency stack is not vendored."
                ),
            ),
            SotaTargetRecord(
                target_id="gnn-benchmark-phm2012-fc-stgnn",
                method_name="GNN RUL Benchmarking FC-STGNN",
                dataset_name="PHM2012",
                condition_name="benchmark_protocol",
                metric_name="rmse",
                target_value=0.1090,
                metric_direction="lower",
                source_type="open_source_benchmark_repo",
                source_url="https://github.com/Frank-Wang-oss/GNN_RUL_Benchmarking",
                source_commit="9325667ed34976452e9323728e33a29fe0f98b5e",
                split_description=(
                    "Repository PHM2012 benchmark protocol using the GNN_RUL_Benchmarking preprocessed split; "
                    "command records --dataset PHM2012 --num_runs 5 and must be rerun before publication."
                ),
                reproducibility_status="open_source_external_env_required",
                license_name="not_declared_on_github_page",
                run_command="python main.py --experiment_description exp1 --run_description run_1 --GNN_method FC_STGNN --dataset PHM2012 --num_runs 5",
                notes="Used as an open-source strong-benchmark route; exact PHM2012 row must be regenerated before final publication.",
            ),
            SotaTargetRecord(
                target_id="weibull-kiml-femto-rmse",
                method_name="Weibull KIML",
                dataset_name="PRONOSTIA",
                condition_name="femto_bearing",
                metric_name="normalized_rmse",
                target_value=0.1771,
                metric_direction="lower",
                source_type="open_source_repo_with_paper",
                source_url="https://github.com/tvhahn/weibull-knowledge-informed-ml",
                source_commit="c430d4b710450a1661e528675a6c1ccc64bc98e2",
                split_description=(
                    "Repository FEMTO/PRONOSTIA train-test protocol driven by make train_femto and summary CSV generation; "
                    "target is from repository summary, not this project split."
                ),
                reproducibility_status="open_source_external_env_required",
                license_name="MIT",
                run_command="make train_femto && make summarize_femto_models && make figures_results",
                notes="Open-source physics/reliability prior reference; target value is from repository summary CSV reported by survey.",
            ),
            SotaTargetRecord(
                target_id="jiang-xjtu-c1-feature-transformer-rmse",
                method_name="Feature-Transformer",
                dataset_name="XJTU-SY",
                condition_name="condition_1_35Hz12kN",
                metric_name="normalized_rmse",
                target_value=0.0885,
                metric_direction="lower",
                source_type="paper_reference_with_local_reimplementation",
                source_url="https://www.mdpi.com/1424-8220/26/5/1578",
                source_commit="local-implementation",
                split_description="Bearing1_1, Bearing1_2, Bearing1_4, Bearing1_5 train; Bearing1_3 test",
                reproducibility_status="local_reimplementation_repeated_configs",
                license_name="paper",
                run_command="uv run python scripts/run_formal_paper_reproductions.py",
                notes="Used as a no-cherry-pick repeated local strong baseline; mean gap must be reported.",
            ),
            SotaTargetRecord(
                target_id="jiang-xjtu-c1-xlstm-rmse",
                method_name="XLSTM-Transformer",
                dataset_name="XJTU-SY",
                condition_name="condition_1_35Hz12kN",
                metric_name="normalized_rmse",
                target_value=0.0583,
                metric_direction="lower",
                source_type="paper_reference_with_local_reimplementation",
                source_url="https://www.mdpi.com/1424-8220/26/5/1578",
                source_commit="local-implementation",
                split_description="Bearing1_1, Bearing1_2, Bearing1_4, Bearing1_5 train; Bearing1_3 test",
                reproducibility_status="local_reimplementation_repeated_configs",
                license_name="paper",
                run_command="uv run python scripts/run_formal_paper_reproductions.py",
                notes="Best local run is close; repeated mean still requires optimization.",
            ),
            SotaTargetRecord(
                target_id="rgpd-phm2012-reference-rmse",
                method_name="RGPD",
                dataset_name="PHM2012",
                condition_name="official_split_reference",
                metric_name="rmse",
                target_value=0.0778,
                metric_direction="lower",
                source_type="paper_reference_only",
                source_url="https://arxiv.org/html/2507.09766v2",
                source_commit="paper-reference-only",
                split_description="Published PHM2012 comparison table; no verified source repository found in this run",
                reproducibility_status="reference_only_not_acceptance_target",
                license_name="paper",
                run_command="not available",
                notes="Reference ceiling only; not counted as open-source acceptance target.",
            ),
        ]
        target_frame = pd.DataFrame([target.to_dict() for target in targets], columns=TARGET_COLUMNS)
        validate_target_frame(target_frame)
        return target_frame

    def build_reproduction_summary(self, target_frame: pd.DataFrame) -> pd.DataFrame:
        """
        build local reproduction summary from existing formal evidence.

        Parameters
        ----------
        target_frame : pd.DataFrame
            target table

        Returns
        -------
        pd.DataFrame
            reproduction summary
        """

        validate_target_frame(target_frame)
        targets = {
            str(row["target_id"]): SotaTargetRecord(**{column: row[column] for column in TARGET_COLUMNS})
            for _, row in target_frame.iterrows()
        }
        records: list[SotaReproductionRecord] = []
        records.extend(self._build_jiang_records(targets))
        rulsurv_records = self._load_rulsurv_port_records(targets)
        records.extend(rulsurv_records)
        records.extend(
            self._build_external_pending_records(
                targets,
                reproduced_target_ids={record.target_id for record in rulsurv_records},
            )
        )
        reproduction_frame = pd.DataFrame([record.to_dict() for record in records], columns=REPRODUCTION_COLUMNS)
        validate_reproduction_frame(reproduction_frame, min_run_count=1)
        return reproduction_frame

    def build_metric_driven_summary(self, reproduction_frame: pd.DataFrame) -> pd.DataFrame:
        """
        build metric-driven comparison summary from reproduction records.

        Parameters
        ----------
        reproduction_frame : pd.DataFrame
            SOTA reproduction table

        Returns
        -------
        pd.DataFrame
            compact metric-driven comparison table
        """

        validate_reproduction_frame(reproduction_frame, min_run_count=1)
        records: list[dict[str, object]] = []
        for row in reproduction_frame.to_dict("records"):
            records.append(
                {
                    "experiment_name": row["experiment_name"],
                    "dataset_name": row["dataset_name"],
                    "condition_name": row["condition_name"],
                    "feature_backend": "formal_19_feature_sequence",
                    "model_backend": row["local_method_name"],
                    "target_method": row["method_name"],
                    "metric_name": row["metric_name"],
                    "target_value": row["target_value"],
                    "local_value": row["local_value"],
                    "local_mean": row["local_mean"],
                    "local_std": row["local_std"],
                    "gap_percent": row["gap_percent"],
                    "mean_gap_percent": row["mean_gap_percent"],
                    "run_count": row["run_count"],
                    "prediction_count": row["prediction_count"],
                    "status": row["status"],
                    "notes": row["notes"],
                }
            )
        return pd.DataFrame.from_records(records)

    def write_artifacts(self) -> dict[str, str]:
        """
        write SOTA target and reproduction summary artifacts.

        Returns
        -------
        dict[str, str]
            output paths
        """

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        targets = self.default_targets()
        reproduction = self.build_reproduction_summary(targets)
        metric_driven = self.build_metric_driven_summary(reproduction)

        target_path = self.evidence_dir / "open_source_sota_targets.csv"
        reproduction_path = self.evidence_dir / "open_source_sota_reproduction_summary.csv"
        metric_driven_path = self.evidence_dir / "metric_driven_comparison_summary.csv"
        targets.to_csv(target_path, index=False)
        reproduction.to_csv(reproduction_path, index=False)
        metric_driven.to_csv(metric_driven_path, index=False)

        manifest_path = self.evidence_dir / "open_source_sota_manifest.json"
        manifest = {
            "target_path": self._display_path(target_path),
            "reproduction_path": self._display_path(reproduction_path),
            "metric_driven_path": self._display_path(metric_driven_path),
            "pass_count": int((reproduction["status"] == "PASS").sum()),
            "needs_optimization_count": int((reproduction["status"] == "NEEDS_OPTIMIZATION").sum()),
            "blocked_count": int(reproduction["status"].astype(str).str.startswith("BLOCKED").sum()),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = self._display_path(manifest_path)
        return {key: str(value) for key, value in manifest.items()}

    def _build_jiang_records(self, targets: dict[str, SotaTargetRecord]) -> list[SotaReproductionRecord]:
        records: list[SotaReproductionRecord] = []
        comparison_frames = self._load_xlstm_comparison_frames()
        for target_id, local_method_name in [
            ("jiang-xjtu-c1-feature-transformer-rmse", "Feature-Transformer"),
            ("jiang-xjtu-c1-xlstm-rmse", "XLSTM-Transformer"),
        ]:
            target = targets[target_id]
            rows = []
            for evidence_path, comparison_frame in comparison_frames:
                matched = comparison_frame[
                    (comparison_frame["dataset_name"] == target.dataset_name)
                    & (comparison_frame["condition_name"] == target.condition_name)
                    & (comparison_frame["model_name"] == local_method_name)
                ]
                if not matched.empty:
                    row = matched.iloc[0].copy()
                    row["evidence_path"] = str(evidence_path)
                    rows.append(row)
            if not rows:
                records.append(self._blocked_record(target, local_method_name=local_method_name, notes="No matching formal reproduction rows found."))
                continue
            metric_values = np.asarray([float(row[target.metric_name]) for row in rows], dtype=float)
            prediction_count = int(np.median([int(row["prediction_count"]) for row in rows]))
            best_value = float(metric_values.min()) if target.metric_direction == "lower" else float(metric_values.max())
            mean_value = float(metric_values.mean())
            std_value = float(metric_values.std(ddof=0))
            mean_gap = float((mean_value - target.target_value) / target.target_value * 100.0)
            status = "PASS" if mean_gap <= 25.0 else "NEEDS_OPTIMIZATION"
            records.append(
                SotaReproductionRecord.from_target(
                    target,
                    experiment_name=f"{target.dataset_name}-{target.condition_name}-{local_method_name}-repeated-formal",
                    local_method_name=local_method_name,
                    local_value=best_value,
                    local_mean=mean_value,
                    local_std=std_value,
                    run_count=len(rows),
                    seeds="formal_50ep,relative,time_index,time_index_seed0",
                    prediction_count=prediction_count,
                    evidence_path=";".join(self._display_path(Path(row["evidence_path"])) for row in rows),
                    status=status,
                    notes=(
                        "Repeated formal evidence uses existing 50 epoch run directories. "
                        "Status is based on mean gap, while local_value stores the best observed repeated value."
                    ),
                )
            )
        return records

    def _load_rulsurv_port_records(self, targets: dict[str, SotaTargetRecord]) -> list[SotaReproductionRecord]:
        summary_path = self.evidence_dir / "rulsurv_rsf_port" / "rulsurv_rsf_port_summary.csv"
        if not summary_path.exists():
            return []
        summary_frame = pd.read_csv(summary_path)
        records: list[SotaReproductionRecord] = []
        for row in summary_frame.to_dict("records"):
            target_id = str(row["target_id"])
            if target_id not in targets:
                continue
            records.append(
                SotaReproductionRecord(
                    target_id=target_id,
                    experiment_name=str(row["experiment_name"]),
                    method_name=str(row["method_name"]),
                    local_method_name=str(row["local_method_name"]),
                    dataset_name=str(row["dataset_name"]),
                    condition_name=str(row["condition_name"]),
                    metric_name=str(row["metric_name"]),
                    target_value=float(row["target_value"]),
                    local_value=float(row["local_value"]),
                    local_mean=float(row["local_mean"]),
                    local_std=float(row["local_std"]),
                    gap_percent=float(row["gap_percent"]),
                    mean_gap_percent=float(row["mean_gap_percent"]),
                    metric_direction=str(row["metric_direction"]),
                    run_count=int(row["run_count"]),
                    seeds=str(row["seeds"]),
                    prediction_count=int(row["prediction_count"]),
                    source_url=str(row["source_url"]),
                    source_commit=str(row["source_commit"]),
                    split_description=str(row["split_description"]),
                    evidence_path=str(row["evidence_path"]),
                    status=str(row["status"]),
                    notes=str(row["notes"]),
                )
            )
        return records

    def _build_external_pending_records(
        self,
        targets: dict[str, SotaTargetRecord],
        *,
        reproduced_target_ids: set[str] | None = None,
    ) -> list[SotaReproductionRecord]:
        reproduced_target_ids = reproduced_target_ids or set()
        records: list[SotaReproductionRecord] = []
        for target_id in [
            "autorul-pronostia-femto-rmse",
            "rulsurv-xjtu-high-rsf-true-mae",
            "gnn-benchmark-phm2012-fc-stgnn",
            "weibull-kiml-femto-rmse",
            "rgpd-phm2012-reference-rmse",
        ]:
            if target_id in reproduced_target_ids:
                continue
            target = targets[target_id]
            status = "BLOCKED_EXTERNAL_ENV" if target.reproducibility_status != "open_source_data_layer" else "REFERENCE_ONLY"
            records.append(
                self._blocked_record(
                    target,
                    local_method_name="not_run_in_project_environment",
                    notes=(
                        "Target locked for SOTA evidence. Local reproduction is not claimed because the external "
                        "environment, dependency stack, or metric protocol differs from this project."
                    ),
                    status=status,
                )
            )
        return records

    def _blocked_record(
        self,
        target: SotaTargetRecord,
        *,
        local_method_name: str,
        notes: str,
        status: str = "BLOCKED",
    ) -> SotaReproductionRecord:
        return SotaReproductionRecord.from_target(
            target,
            experiment_name=f"{target.dataset_name}-{target.condition_name}-{target.method_name}-target-only",
            local_method_name=local_method_name,
            local_value=float("nan"),
            local_mean=float("nan"),
            local_std=float("nan"),
            run_count=1,
            seeds="not_run",
            prediction_count=0,
            evidence_path="not_available",
            status=status,
            notes=notes,
        )

    def _load_xlstm_comparison_frames(self) -> list[tuple[Path, pd.DataFrame]]:
        patterns = [
            "tmp/formal_paper_reproductions_50ep/paper_xlstm_transformer/comparison_metrics.csv",
            "tmp/formal_paper_reproductions_50ep_relative/paper_xlstm_transformer/comparison_metrics.csv",
            "tmp/formal_paper_reproductions_50ep_time_index/paper_xlstm_transformer/comparison_metrics.csv",
            "tmp/formal_paper_reproductions_50ep_time_index_seed0/paper_xlstm_transformer/comparison_metrics.csv",
        ]
        frames: list[tuple[Path, pd.DataFrame]] = []
        for pattern in patterns:
            path = self.project_root / pattern
            if path.exists():
                frame = pd.read_csv(path)
                if "r2" not in frame.columns and "r2_score" in frame.columns:
                    frame["r2"] = frame["r2_score"]
                frames.append((Path(pattern), frame))
        summary_path = self.evidence_dir / "xlstm_transformer_comparison_summary.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            if "r2" not in frame.columns and "r2_score" in frame.columns:
                frame["r2"] = frame["r2_score"]
            frames.append((Path("docs/reproduction-evidence/xlstm_transformer_comparison_summary.csv"), frame))
        return frames

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path)

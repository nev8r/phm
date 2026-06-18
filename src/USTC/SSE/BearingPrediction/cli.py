"""
Command line interface for bearing PHM workflows

this file is for defining unified analyze, train, benchmark, and report commands

created by zy

copyright USTC

2026
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    build_domain_feature_names,
    build_dataset_cards,
    build_sample_feature_table,
    compute_feature_analysis,
    compute_tsfresh_audit,
    render_feature_figures,
    render_model_architecture_diagrams,
    render_tsfresh_audit_figures,
    task_relationship_summary,
    write_json,
)
from .workflow import run_benchmark as run_benchmark_workflow
from .workflow import build_or_load_feature_cache
from .workflow import evaluate_saved_training_run
from .workflow import predict_feature_csv_with_run
from .workflow import run_paper_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phm",
        description="Bearing PHM analysis, training, benchmark, and reporting CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="run dataset and feature analysis")
    analyze.add_argument("--task", choices=["rul", "fault", "all"], default="all")
    analyze.add_argument("--feature-set", choices=["domain", "tsfresh", "advanced"], default="domain")
    analyze.add_argument("--full", action="store_true", help="use full cached/data-backed analysis when available")
    analyze.add_argument("--sample", action="store_true", help="run a small deterministic smoke analysis")
    analyze.add_argument("--output-dir", default="outputs/runs", help="directory for run artifacts")
    analyze.add_argument("--run-dir", default=None, help="explicit directory for this analysis run")

    cache = subparsers.add_parser("cache", help="build or inspect paper feature caches")
    cache.add_argument("--task", choices=["rul", "fault", "all"], default="all")
    cache.add_argument("--force", action="store_true", help="rebuild cache even when it already exists")
    cache.add_argument("--phm-root", default=None, help="PHM2012 root directory")
    cache.add_argument("--xjtu-root", default=None, help="XJTU-SY root directory")
    cache.add_argument("--output-dir", default="outputs/gui/cache", help="directory for cache job artifacts")
    cache.add_argument("--run-dir", default=None, help="explicit directory for this cache job")

    train = subparsers.add_parser("train", help="train a paper reproduction model")
    train.add_argument("--task", choices=["rul", "fault"], required=True)
    train.add_argument("--preset", choices=["paper", "smoke"], default="paper")
    train.add_argument("--full", action="store_true", help="use full dataset/cache")
    train.add_argument("--sample", action="store_true", help="run a short smoke training job")
    train.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    train.add_argument("--output-dir", default="outputs/runs", help="directory for run artifacts")
    train.add_argument("--run-dir", default=None, help="explicit directory for this training run")

    benchmark = subparsers.add_parser("benchmark", help="compare baselines with shared splits and metrics")
    benchmark.add_argument("--task", choices=["rul", "fault", "all"], default="all")
    benchmark.add_argument("--baselines", default="all", help="comma separated baseline list or all")
    benchmark.add_argument("--full", action="store_true", help="run full baseline matrix")
    benchmark.add_argument("--sample", action="store_true", help="run short smoke baselines")
    benchmark.add_argument("--output-dir", default="outputs/runs", help="directory for run artifacts")
    benchmark.add_argument("--run-dir", default=None, help="explicit directory for this benchmark run")

    report = subparsers.add_parser("report", help="summarize a previous PHM run")
    report.add_argument("--run", required=True, help="path to outputs/runs/<run_id>")

    evaluate = subparsers.add_parser("evaluate", help="reload a training run and evaluate the fixed test split")
    evaluate.add_argument("--run", required=True, help="training run directory")
    evaluate.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    evaluate.add_argument("--output-dir", default=None, help="directory for evaluation artifacts")

    predict = subparsers.add_parser("predict", help="run feature CSV inference with a saved training run")
    predict.add_argument("--run", required=True, help="training run directory")
    predict.add_argument("--csv", required=True, help="feature CSV path")
    predict.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    predict.add_argument("--output-dir", default=None, help="directory for prediction artifacts")

    gui = subparsers.add_parser("gui", help="launch the local Streamlit experiment workbench")
    gui.add_argument("--port", type=int, default=8501, help="Streamlit server port")
    gui.add_argument("--host", default="localhost", help="Streamlit bind address")
    gui.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="run Streamlit headless")
    return parser


def _new_run_dir(output_dir: Path | str, command: str, task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = task if task != "all" else "all"
    run_dir = Path(output_dir) / f"{timestamp}_{command}_{suffix}"
    counter = 1
    while run_dir.exists():
        run_dir = Path(output_dir) / f"{timestamp}_{command}_{suffix}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def _task_list(task: str) -> list[str]:
    return ["rul", "fault"] if task == "all" else [task]


def _compact_feature_analysis_for_json(analysis: dict[str, Any]) -> dict[str, Any]:
    compact = dict(analysis)
    heatmap = np.asarray(compact.pop("correlation_heatmap"), dtype=np.float32)
    compact["correlation_heatmap_shape"] = list(heatmap.shape)
    compact["correlation_heatmap_note"] = "full matrix is rendered to the heatmap figure; JSON keeps Top-K ranks only"
    return compact


def _load_full_feature_table(
    task: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any], np.ndarray | None, np.ndarray | None]:
    cache_dir = Path("cache/paper_features")
    candidates = {
        "rul": [
            cache_dir / "phm2012_rul_fft256_full.npz",
            cache_dir / "phm2012_rul_fft256_paper_full.npz",
        ],
        "fault": [
            cache_dir / "xjtu_binary_fault_diagnosis_fft256_full.npz",
            cache_dir / "xjtu_fault_fft256_full.npz",
        ],
    }[task]
    for path in candidates:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        targets_key = "targets" if "targets" in data.files else "labels"
        target = data[targets_key].astype(np.float32)
        metadata: dict[str, Any] = {"source": str(path), "mode": "full-cache"}
        if "metadata" in data.files:
            try:
                metadata.update(json.loads(str(data["metadata"].item())))
            except (TypeError, ValueError, AttributeError):
                metadata["metadata_parse_error"] = True
        if "feature_names" in data.files:
            feature_names = [str(item) for item in data["feature_names"].tolist()]
        else:
            fft_bins = int(metadata.get("fft_bins", 256))
            include_handcrafted = bool(metadata.get("include_handcrafted", True))
            per_channel = fft_bins + (20 if include_handcrafted else 0)
            channel_count = max(1, features.shape[1] // max(1, per_channel))
            feature_names = build_domain_feature_names(
                fft_bins=fft_bins,
                include_handcrafted=include_handcrafted,
                channel_count=channel_count,
            )
            if len(feature_names) != features.shape[1]:
                feature_names = [f"feature_{index}" for index in range(features.shape[1])]
        ids = data["bearing_names"].astype(str) if "bearing_names" in data.files else None
        times = data["file_indices"].astype(int) if "file_indices" in data.files else None
        return features, target, feature_names, metadata, ids, times
    features, target, names = build_sample_feature_table(task)
    return features, target, names, {
        "source": "sample-fallback",
        "mode": "sample",
        "warning": "full cache was not found, deterministic sample data was used",
    }, None, None


def _run_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _new_run_dir(args.output_dir, "analyze", args.task)
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "command": "analyze",
        "task": args.task,
        "feature_set": args.feature_set,
        "dataset_cards": build_dataset_cards(),
        "task_relationship": task_relationship_summary(),
        "analyses": {},
        "figures": {},
    }

    render_paths = render_model_architecture_diagrams(figures_dir)
    metrics["figures"]["model_architectures"] = render_paths

    for task in _task_list(args.task):
        if args.sample or not args.full:
            features, target, feature_names = build_sample_feature_table(task)
            source_meta = {"source": "deterministic-sample", "mode": "sample"}
            ids = None
            times = None
        else:
            features, target, feature_names, source_meta, ids, times = _load_full_feature_table(task)
        analysis = compute_feature_analysis(features, target, feature_names, task=task)
        tsfresh_mode = "efficient" if args.feature_set == "advanced" else "minimal"
        tsfresh_max_features = 6 if tsfresh_mode == "efficient" else 12
        tsfresh_audit = compute_tsfresh_audit(
            features,
            target,
            feature_names,
            ids=ids,
            times=times,
            mode=tsfresh_mode,
            max_domain_features=tsfresh_max_features,
        )
        feature_figures = render_feature_figures(figures_dir, analysis, feature_names, prefix=task)
        tsfresh_figures = {}
        if args.feature_set in {"tsfresh", "advanced"}:
            tsfresh_figures = render_tsfresh_audit_figures(figures_dir, tsfresh_audit, prefix=task)
        metrics["analyses"][task] = {
            "source": source_meta,
            "feature_names": feature_names,
            "summary": _compact_feature_analysis_for_json(analysis),
            "tsfresh_audit": tsfresh_audit,
        }
        metrics["figures"][task] = {
            "domain": feature_figures,
            "tsfresh_audit": tsfresh_figures,
        }

    config = {
        "command": "analyze",
        "task": args.task,
        "feature_set": args.feature_set,
        "full": bool(args.full),
        "sample": bool(args.sample),
        "output_dir": str(args.output_dir),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _new_run_dir(args.output_dir, "train", args.task)
    run_paper_training(
        task=args.task,
        preset=args.preset,
        sample=bool(args.sample or not args.full),
        device_name=args.device,
        run_dir=run_dir,
    )
    print(f"run_dir={run_dir}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _new_run_dir(args.output_dir, "benchmark", args.task)
    run_benchmark_workflow(
        task=args.task,
        baselines=args.baselines,
        sample=bool(args.sample or not args.full),
        run_dir=run_dir,
    )
    print(f"run_dir={run_dir}")
    return 0


def _run_cache(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else _new_run_dir(args.output_dir, "cache", args.task)
    tasks = _task_list(args.task)
    results = {}
    for task in tasks:
        results[task] = build_or_load_feature_cache(
            task,
            force=bool(args.force),
            phm_root=args.phm_root,
            xjtu_root=args.xjtu_root,
        )
    config = {
        "command": "cache",
        "task": args.task,
        "force": bool(args.force),
        "phm_root": args.phm_root,
        "xjtu_root": args.xjtu_root,
    }
    metrics = {"command": "cache", "task": args.task, "results": results}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    print(f"run_dir={run_dir}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    run = Path(args.run)
    if not run.exists():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    summary = run / "summary.txt"
    metrics = run / "metrics.json"
    text = f"PHM run: {run}\nmetrics: {metrics if metrics.exists() else 'missing'}\n"
    summary.write_text(text, encoding="utf-8")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_saved_training_run(
        args.run,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"output_dir={result['output_dir']}")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    result = predict_feature_csv_with_run(
        args.run,
        args.csv,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(f"output_dir={result['output_dir']}")
    return 0


def _run_gui(args: argparse.Namespace) -> int:
    gui_path = Path(__file__).with_name("gui.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(gui_path),
        "--server.port",
        str(args.port),
        "--server.address",
        str(args.host),
        "--server.headless",
        "true" if args.headless else "false",
    ]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "cache":
        return _run_cache(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "gui":
        return _run_gui(args)
    raise ValueError(f"unknown command: {args.command}")


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

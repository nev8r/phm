"""
Run sktime RUL baseline

this script runs sktime panel regressors for XJTU RUL

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from USTC.SSE.BearingPrediction.experiments.metric_rul_baselines import (
    cli_config,
    exit_with_dependency_message,
    print_paths,
    run_sktime_rul_baseline,
)


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Run sktime Rocket and TimeSeriesForest RUL baselines.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--xjtu-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--downsample-points", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """
    run script
    """

    args = parse_args()
    config = cli_config(args.project_root, args.xjtu_root, args.output_dir, args.seeds, args.downsample_points)
    try:
        paths = run_sktime_rul_baseline(config)
    except RuntimeError as exc:
        exit_with_dependency_message(exc)
    print_paths(paths)


if __name__ == "__main__":
    main()

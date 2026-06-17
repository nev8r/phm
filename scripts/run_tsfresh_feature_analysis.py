"""
Run tsfresh feature relevance analysis

this script generates tsfresh feature relevance evidence for XJTU RUL

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
    run_tsfresh_feature_analysis,
)


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Run tsfresh feature relevance analysis on XJTU-SY condition 1.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--xjtu-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--downsample-points", type=int, default=256)
    parser.add_argument("--tsfresh-configs", nargs="+", default=["minimal", "efficient"])
    return parser.parse_args()


def main() -> None:
    """
    run script
    """

    args = parse_args()
    config = cli_config(
        args.project_root,
        args.xjtu_root,
        args.output_dir,
        args.seeds,
        args.downsample_points,
        args.tsfresh_configs,
    )
    try:
        paths = run_tsfresh_feature_analysis(config)
    except RuntimeError as exc:
        exit_with_dependency_message(exc)
    print_paths(paths)


if __name__ == "__main__":
    main()

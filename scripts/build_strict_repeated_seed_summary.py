"""
Build strict repeated seed summary

this script writes same-config repeated seed evidence for formal RUL baselines

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from USTC.SSE.BearingPrediction.experiments.metric_rul_baselines import (
    build_strict_repeated_seed_summary,
    print_paths,
)


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Build strict same-config repeated seed summary.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--xjtu-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202601, 202602, 202603])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--max-samples-per-entity",
        type=int,
        default=0,
        help="Optional cap for debugging. The default 0 uses all available snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    """
    run script
    """

    args = parse_args()
    paths = build_strict_repeated_seed_summary(
        args.project_root.resolve(),
        args.output_dir.resolve() if args.output_dir else None,
        xjtu_root=args.xjtu_root.resolve() if args.xjtu_root else None,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        max_samples_per_entity=args.max_samples_per_entity or None,
    )
    print_paths(paths)


if __name__ == "__main__":
    main()

"""
Open-source SOTA evidence script

this script writes SOTA target and reproduction summary artifacts

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from USTC.SSE.BearingPrediction.experiments.sota import SotaEvidenceBuilder


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Build open-source SOTA evidence CSV files.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root path",
    )
    return parser.parse_args()


def main() -> None:
    """
    run evidence builder
    """

    args = parse_args()
    manifest = SotaEvidenceBuilder(args.project_root).write_artifacts()
    for key, value in manifest.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


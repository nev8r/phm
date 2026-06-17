"""
Run RULSurv RSF port reproduction

this script is for generating open-source SOTA reproduction evidence

created by zyj

copyright USTC

2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from USTC.SSE.BearingPrediction.experiments.sota.sota_adapters import RulSurvRsfPortAdapter, RulSurvRsfPortConfig


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Run the RULSurv RSF port on XJTU-SY condition 1.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--xjtu-root",
        type=Path,
        default=None,
        help="Path to XJTU-SY_Bearing_Datasets; defaults to data/external/xjtu/extracted/XJTU-SY_Bearing_Datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evidence output directory; defaults to docs/reproduction-evidence/rulsurv_rsf_port.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """
    run adapter and print artifact paths
    """

    args = parse_args()
    project_root = args.project_root.resolve()
    xjtu_root = args.xjtu_root or project_root / "data" / "external" / "xjtu" / "extracted" / "XJTU-SY_Bearing_Datasets"
    output_dir = args.output_dir or project_root / "docs" / "reproduction-evidence" / "rulsurv_rsf_port"
    config = RulSurvRsfPortConfig(
        xjtu_root=xjtu_root,
        output_dir=output_dir,
        seeds=tuple(args.seeds),
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
    )
    paths = RulSurvRsfPortAdapter(config).run()
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

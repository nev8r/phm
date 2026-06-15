"""
Run formal real-data RUL paper reproductions.

This script is the non-demo entry point for final paper reproduction
evidence. It requires official-scale real datasets under data/external
unless explicit real roots are provided.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from USTC.SSE.BearingPrediction.examples import run_formal_paper_reproductions


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser = argparse.ArgumentParser(description="Run formal real-data RUL paper reproductions.")
    parser.add_argument("--output-root", type=Path, default=Path("tmp/formal_paper_reproductions"))
    parser.add_argument("--xjtu-root", type=Path, default=None)
    parser.add_argument("--phm2012-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cnn-max-samples", type=int, default=256)
    parser.add_argument("--xlstm-max-samples", type=int, default=256)
    parser.add_argument("--profile", default="formal", choices=["formal", "smoke"])
    return parser.parse_args()


def main() -> None:
    """
    run both formal paper reproductions
    """

    args = parse_args()
    os.environ["BEARING_EXAMPLE_OUTPUT_ROOT"] = str(args.output_root)
    os.environ["BEARING_EXAMPLE_EPOCHS"] = str(args.epochs)
    os.environ["BEARING_EXAMPLE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["BEARING_EXAMPLE_LOSS"] = "mse"
    os.environ["BEARING_FORMAL_TARGET_MODE"] = "entity_relative"
    os.environ["BEARING_REPRODUCTION_PROFILE"] = args.profile
    os.environ["BEARING_FORMAL_CNN_MAX_SAMPLES"] = str(args.cnn_max_samples)
    os.environ["BEARING_EXAMPLE_MAX_SAMPLES"] = str(args.xlstm_max_samples)

    result = run_formal_paper_reproductions(
        xjtu_root=args.xjtu_root,
        phm2012_root=args.phm2012_root,
        cnn_max_samples_per_entity=args.cnn_max_samples,
        xlstm_max_samples_per_entity=args.xlstm_max_samples,
        profile=args.profile,
    )
    print(f"status={result['status']}")
    print(f"summary_path={result['summary_path']}")
    print(f"cnn_comparison={result['cnn_lstm_attention']['comparison_path']}")
    print(f"cnn_paper_reference={result['cnn_lstm_attention']['paper_reference_path']}")
    print(f"xlstm_comparison={result['xlstm_transformer']['comparison_path']}")
    print(f"xlstm_paper_reference={result['xlstm_transformer']['paper_reference_path']}")


if __name__ == "__main__":
    main()

"""Command-line entry point for the published experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DatasetPaths
from .experiment import RUNNERS, ExperimentConfig, run_selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the AE-ngComms experimental sweeps.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=tuple(RUNNERS),
        default=list(RUNNERS),
        help="Experiments to run (default: all published experiments).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Caples_Lake_N7_2014_2017.csv"),
        help="Path to the Caples Lake CSV subset.",
    )
    parser.add_argument(
        "--derived-data-dir",
        type=Path,
        default=Path("data/derived"),
        help="Cache directory for deterministic train/validation/test splits.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/reproduced"),
        help="Output directory; published tables are never overwritten by default.",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda, or mps (default: auto).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one tiny configuration per selected sweep as a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace result tables that already exist in --results-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        dataset=DatasetPaths(
            raw=args.data.resolve(),
            derived_dir=args.derived_data_dir.resolve(),
        ),
        results_dir=args.results_dir.resolve(),
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )
    if args.quick:
        config = config.quick()
    run_selected(config, args.sweeps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

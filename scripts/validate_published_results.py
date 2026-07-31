#!/usr/bin/env python3
"""Validate the archived tables against the values reported in the article."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

EXPECTED_ROWS = {
    "sweep_b": 500,
    "sweep_c": 2000,
    "sweep_d": 700,
    "sweep_rice": 383,
    "sweep_gorilla": 383,
}


def load(results_dir: Path, stem: str) -> list[dict[str, str]]:
    with (results_dir / f"{stem}_results.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def selected_mean(
    rows: list[dict[str, str]],
    column: str,
    **conditions: str,
) -> float:
    selected = [
        float(row[column])
        for row in rows
        if all(row[key].lower() == value.lower() for key, value in conditions.items())
    ]
    if not selected:
        raise AssertionError(f"no rows matched {conditions}")
    return statistics.fmean(selected)


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} differs from expected {expected}")


def validate(results_dir: Path) -> None:
    tables = {stem: load(results_dir, stem) for stem in EXPECTED_ROWS}
    for stem, expected in EXPECTED_ROWS.items():
        actual = len(tables[stem])
        if actual != expected:
            raise AssertionError(f"{stem}: expected {expected} rows, found {actual}")

    sweep_b = tables["sweep_b"]
    sweep_c = tables["sweep_c"]
    sweep_d = tables["sweep_d"]
    rice = tables["sweep_rice"]
    gorilla = tables["sweep_gorilla"]

    close(
        selected_mean(
            sweep_c,
            "mae_mean",
            input_dim="128",
            target_ratio="2",
            symmetric="true",
        ),
        0.84379,
    )
    close(
        selected_mean(
            sweep_c,
            "mae_mean",
            input_dim="128",
            target_ratio="32",
            symmetric="true",
        ),
        2.02788,
    )
    close(
        selected_mean(
            sweep_d,
            "mae_mean",
            latent_dim="2",
            symmetric="true",
        ),
        3.91721,
    )
    close(
        selected_mean(
            sweep_d,
            "mae_mean",
            latent_dim="128",
            symmetric="false",
        ),
        2.11756,
    )
    close(
        selected_mean(
            sweep_b,
            "mae_mean",
            hidden_layers="2",
            symmetric="true",
        ),
        0.80006,
    )
    close(
        selected_mean(
            sweep_b,
            "mae_mean",
            hidden_layers="3",
            symmetric="false",
        ),
        0.77834,
    )
    close(
        selected_mean(rice, "ratio", input_dim="128"),
        4.18964,
    )
    close(
        selected_mean(gorilla, "ratio", input_dim="128"),
        1.13283,
    )
    if any(float(row["mae"]) != 0.0 for row in gorilla):
        raise AssertionError("Gorilla round trips are not bit exact")

    print("Published result tables are internally consistent with the article.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/published"),
    )
    args = parser.parse_args()
    validate(args.results_dir.resolve())


if __name__ == "__main__":
    main()

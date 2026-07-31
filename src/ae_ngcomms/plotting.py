"""Regenerate the three result figures from the archived CSV tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ENERGY_PJ_PER_MAC = 4.6
SYMMETRIC_COLOR = "#0073bc"
ASYMMETRIC_COLOR = "#d9541a"
RICE_COLOR = "#7d2e8e"
GORILLA_COLOR = "#4d994d"


def confidence_interval(values: pd.Series) -> tuple[float, float]:
    """Return mean and the 95% half-width used by the original MATLAB plots."""
    data = values.to_numpy(dtype=float)
    count = len(data)
    mean = float(np.mean(data))
    if count < 2:
        return mean, 0.0
    degrees_of_freedom = count - 1
    half_width = (
        float(t.ppf(0.975, degrees_of_freedom))
        * float(np.std(data, ddof=1))
        / math.sqrt(degrees_of_freedom)
    )
    return mean, half_width


def _series(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.sort(frame[x_column].unique())
    means = []
    intervals = []
    for x_value in x_values:
        mean, interval = confidence_interval(
            frame.loc[frame[x_column] == x_value, y_column]
        )
        means.append(mean)
        intervals.append(interval)
    return x_values, np.asarray(means), np.asarray(intervals)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)


def plot_block_length(sweep: pd.DataFrame, destination: Path) -> None:
    """MAE versus nominal ratio for the four block lengths in sweep C."""
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for axis, symmetric, title in zip(
        axes,
        (True, False),
        ("Symmetric AE", "Asymmetric AE"),
        strict=True,
    ):
        subset = sweep[sweep["symmetric"] == symmetric]
        for input_dim in sorted(subset["input_dim"].unique()):
            rows = subset[subset["input_dim"] == input_dim]
            x_values, means, intervals = _series(rows, "ratio", "mae_mean")
            axis.errorbar(
                x_values,
                means,
                yerr=intervals,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                capsize=2,
                label=f"n={input_dim}",
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Nominal compression ratio", fontsize=8)
        _style_axis(axis)
    axes[0].set_ylabel("MAE (°C)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_compression_baselines(
    autoencoders: pd.DataFrame,
    rice: pd.DataFrame,
    gorilla: pd.DataFrame,
    destination: Path,
) -> None:
    """Compression ratio and MAE for AEs and conventional baselines."""
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    configurations = (
        (True, "Symmetric AE", SYMMETRIC_COLOR, "o", "-"),
        (False, "Asymmetric AE", ASYMMETRIC_COLOR, "s", "--"),
    )
    for symmetric, label, color, marker, line_style in configurations:
        rows = autoencoders[autoencoders["symmetric"] == symmetric]
        for axis, y_column in zip(
            (axes[0, 0], axes[1, 0]),
            ("ratio", "mae_mean"),
            strict=True,
        ):
            x_values, means, intervals = _series(
                rows,
                "latent_dim",
                y_column,
            )
            axis.errorbar(
                x_values,
                means,
                yerr=intervals,
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.2,
                markersize=3.5,
                capsize=2,
                label=label,
            )

    for rows, label, color, marker, line_style in (
        (rice, "Rice-Golomb", RICE_COLOR, "P", "-"),
        (gorilla, "Gorilla", GORILLA_COLOR, "h", "--"),
    ):
        for axis, y_column in zip(
            (axes[0, 1], axes[1, 1]),
            ("ratio", "mae"),
            strict=True,
        ):
            x_values, means, intervals = _series(rows, "input_dim", y_column)
            axis.errorbar(
                x_values,
                means,
                yerr=intervals,
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.2,
                markersize=4,
                capsize=2,
                label=label,
            )

    axes[0, 0].set_title("Autoencoders", fontsize=9)
    axes[0, 1].set_title("Rice-Golomb / Gorilla", fontsize=9)
    axes[0, 0].set_ylabel("Compression ratio", fontsize=8)
    axes[1, 0].set_ylabel("MAE (°C)", fontsize=8)
    axes[1, 0].set_xlabel("Latent dimension", fontsize=8)
    axes[1, 1].set_xlabel("Window size", fontsize=8)
    axes[1, 1].set_ylim(0.0, axes[1, 0].get_ylim()[1])
    for axis in axes.flat:
        _style_axis(axis)

    handles = []
    labels = []
    for axis in (axes[0, 0], axes[0, 1]):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_energy_accuracy(sweep: pd.DataFrame, destination: Path) -> None:
    """Encoder/decoder compute-derived energy and MAE versus depth index."""
    frame = sweep.copy()
    frame["encoder_energy_nj"] = frame["encoder_macs"] * ENERGY_PJ_PER_MAC / 1000
    frame["decoder_energy_nj"] = frame["decoder_macs"] * ENERGY_PJ_PER_MAC / 1000
    figure = plt.figure(figsize=(7.0, 4.5))
    grid = figure.add_gridspec(2, 2)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    )
    for symmetric, label, color in (
        (True, "Symmetric (AE)", SYMMETRIC_COLOR),
        (False, "Asymmetric (AAE)", ASYMMETRIC_COLOR),
    ):
        rows = frame[frame["symmetric"] == symmetric]
        for axis, column in zip(
            axes,
            ("encoder_energy_nj", "decoder_energy_nj", "mae_mean"),
            strict=True,
        ):
            x_values, means, intervals = _series(
                rows,
                "hidden_layers",
                column,
            )
            axis.errorbar(
                x_values,
                means,
                yerr=intervals,
                color=color,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                capsize=2,
                label=label,
            )
    axes[0].set_ylabel("Encoder energy (nJ)", fontsize=8)
    axes[1].set_ylabel("Decoder energy (nJ)", fontsize=8)
    axes[2].set_ylabel("MAE (°C)", fontsize=8)
    axes[2].set_xlabel("Depth index", fontsize=8)
    for axis in axes:
        axis.set_xticks(sorted(frame["hidden_layers"].unique()))
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def generate_all(results_dir: Path, output_dir: Path) -> None:
    """Load the archived tables and write all result plots as vector PDFs."""
    tables = {
        stem: pd.read_csv(results_dir / f"{stem}_results.csv")
        for stem in (
            "sweep_b",
            "sweep_c",
            "sweep_d",
            "sweep_rice",
            "sweep_gorilla",
        )
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_block_length(
        tables["sweep_c"],
        output_dir / "block_length_mae.pdf",
    )
    plot_compression_baselines(
        tables["sweep_d"],
        tables["sweep_rice"],
        tables["sweep_gorilla"],
        output_dir / "compression_baselines.pdf",
    )
    plot_energy_accuracy(
        tables["sweep_b"],
        output_dir / "energy_accuracy.pdf",
    )
    print(f"generated three figures in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate article figures from experiment CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/published"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/reproduced"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    generate_all(args.results_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

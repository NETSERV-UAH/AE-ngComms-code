"""Reproduction of the autoencoder and lossless-codec sweeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .codecs import (
    BitReader,
    best_rice_parameter,
    gorilla_decode,
    gorilla_encode,
    rice_decode,
    rice_encode,
    zigzag_decode,
    zigzag_encode,
)
from .data import (
    AutoencoderWindows,
    DatasetPaths,
    RiceWindows,
    load_windowed_splits,
)
from .models import (
    AsymmetricAutoencoder,
    build_autoencoder,
    linear_macs,
)
from .training import TrainConfig, fit, seed_everything

SWEEP_FILENAMES = {
    "b": "sweep_b_results.csv",
    "c": "sweep_c_results.csv",
    "d": "sweep_d_results.csv",
    "rice": "sweep_rice_results.csv",
    "gorilla": "sweep_gorilla_results.csv",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experimental protocol, with article values as defaults."""

    dataset: DatasetPaths
    results_dir: Path
    seeds: int = 50
    epochs: int = 300
    patience: int = 15
    device: str = "auto"
    training_strides: tuple[int, ...] = (10, 27, 33)
    hidden_activation: str = "elu"
    sweep_b_input_dim: int = 100
    sweep_b_latent_dim: int = 25
    sweep_b_depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    sweep_c_input_dims: tuple[int, ...] = (128, 256, 512, 1024)
    sweep_c_ratios: tuple[int, ...] = (2, 4, 8, 16, 32)
    sweep_c_depth: int = 2
    sweep_d_input_dim: int = 256
    sweep_d_latent_dims: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)
    sweep_d_depth: int = 2
    baseline_input_dims: tuple[int, ...] = (128, 256, 512, 1024)

    def __post_init__(self) -> None:
        if self.seeds <= 0:
            raise ValueError("seeds must be positive")

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def quick(self) -> ExperimentConfig:
        """Return a small smoke-test version of every selected sweep."""
        return replace(
            self,
            seeds=1,
            epochs=min(self.epochs, 2),
            patience=0,
            sweep_b_depths=self.sweep_b_depths[:1],
            sweep_c_input_dims=self.sweep_c_input_dims[:1],
            sweep_c_ratios=self.sweep_c_ratios[:1],
            sweep_d_latent_dims=self.sweep_d_latent_dims[:1],
            baseline_input_dims=self.baseline_input_dims[:1],
        )


def _load_autoencoder_data(
    config: ExperimentConfig,
    input_dim: int,
) -> tuple[AutoencoderWindows, AutoencoderWindows, AutoencoderWindows]:
    train, validation, test = load_windowed_splits(
        config.dataset,
        input_dim,
        kind="autoencoder",
        training_strides=config.training_strides,
    )
    if not all(
        isinstance(split, AutoencoderWindows) for split in (train, validation, test)
    ):
        raise TypeError("autoencoder slicer returned an unexpected data type")
    return train, validation, test


def train_one_configuration(
    training: AutoencoderWindows,
    validation: AutoencoderWindows,
    *,
    input_dim: int,
    latent_dim: int,
    depth_index: int,
    asymmetric: bool,
    seed: int,
    config: ExperimentConfig,
) -> AsymmetricAutoencoder:
    """Train one seeded architecture using the article's hyperparameters."""
    seed_everything(seed)
    model = build_autoencoder(
        input_dim,
        latent_dim,
        depth_index,
        asymmetric=asymmetric,
        hidden_activation=config.hidden_activation,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    training_loader = DataLoader(
        TensorDataset(torch.from_numpy(training.data)),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.data)),
        batch_size=256,
        shuffle=False,
    )
    fit(
        model,
        training_loader,
        validation_loader,
        TrainConfig(
            epochs=config.epochs,
            early_stopping_patience=config.patience,
            device=config.resolved_device,
        ),
    )
    return model


def evaluate_autoencoder(
    model: AsymmetricAutoencoder,
    test: AutoencoderWindows,
    *,
    device: str,
) -> dict[str, float]:
    """Evaluate reconstructed absolute temperatures, one metric per window."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        reconstruction, _ = model(torch.from_numpy(test.data).to(device))
    predicted = reconstruction.cpu().numpy()

    mean_squared_errors = np.empty(len(test), dtype=np.float64)
    mean_absolute_errors = np.empty(len(test), dtype=np.float64)
    for index in range(len(test)):
        expected = test.reconstruct(index, test.data[index])
        actual = test.reconstruct(index, predicted[index])
        error = expected - actual
        mean_squared_errors[index] = np.mean(error**2)
        mean_absolute_errors[index] = np.mean(np.abs(error))
    return {
        "mse_mean": float(mean_squared_errors.mean()),
        "mse_std": float(mean_squared_errors.std()),
        "mse_p95": float(np.percentile(mean_squared_errors, 95)),
        "mae_mean": float(mean_absolute_errors.mean()),
        "mae_std": float(mean_absolute_errors.std()),
    }


def _model_row(
    *,
    model: AsymmetricAutoencoder,
    test: AutoencoderWindows,
    input_dim: int,
    latent_dim: int,
    depth_index: int,
    asymmetric: bool,
    seed: int,
    device: str,
    include_complexity: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symmetric": not asymmetric,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_layers": depth_index,
        "seed": seed,
    }
    if include_complexity:
        row.update(
            encoder_macs=linear_macs(model.encoder),
            decoder_macs=linear_macs(model.decoder),
            encoder_params=sum(
                parameter.numel() for parameter in model.encoder.parameters()
            ),
            decoder_params=sum(
                parameter.numel() for parameter in model.decoder.parameters()
            ),
        )
    row.update(extra or {})
    row.update(evaluate_autoencoder(model, test, device=device))
    return row


def _train_pair(
    config: ExperimentConfig,
    training: AutoencoderWindows,
    validation: AutoencoderWindows,
    test: AutoencoderWindows,
    *,
    input_dim: int,
    latent_dim: int,
    depth_index: int,
    seed: int,
    include_complexity: bool = False,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for asymmetric in (False, True):
        label = "AAE" if asymmetric else "AE"
        print(
            f"{label}: n={input_dim}, m={latent_dim}, depth={depth_index}, seed={seed}"
        )
        model = train_one_configuration(
            training,
            validation,
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth_index=depth_index,
            asymmetric=asymmetric,
            seed=seed,
            config=config,
        )
        rows.append(
            _model_row(
                model=model,
                test=test,
                input_dim=input_dim,
                latent_dim=latent_dim,
                depth_index=depth_index,
                asymmetric=asymmetric,
                seed=seed,
                device=config.resolved_device,
                include_complexity=include_complexity,
                extra=extra,
            )
        )
    return rows


def run_sweep_b(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Vary branch depth at fixed ``n=100`` and ``m=25``."""
    input_dim = config.sweep_b_input_dim
    latent_dim = config.sweep_b_latent_dim
    training, validation, test = _load_autoencoder_data(config, input_dim)
    rows: list[dict[str, Any]] = []
    for depth_index in config.sweep_b_depths:
        for seed in range(config.seeds):
            rows.extend(
                _train_pair(
                    config,
                    training,
                    validation,
                    test,
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    depth_index=depth_index,
                    seed=seed,
                    include_complexity=True,
                    extra={
                        # Kept for compatibility with the archived result table.
                        "ratio": input_dim / (latent_dim + 3),
                        "nominal_ratio": input_dim / latent_dim,
                        "ratio_with_side_information": input_dim / (latent_dim + 3),
                    },
                )
            )
    return rows


def run_sweep_c(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Vary block length and nominal compression ratio."""
    rows: list[dict[str, Any]] = []
    for input_dim in config.sweep_c_input_dims:
        training, validation, test = _load_autoencoder_data(config, input_dim)
        for target_ratio in config.sweep_c_ratios:
            latent_dim = max(1, round(input_dim / target_ratio))
            nominal_ratio = input_dim / latent_dim
            for seed in range(config.seeds):
                rows.extend(
                    _train_pair(
                        config,
                        training,
                        validation,
                        test,
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        depth_index=config.sweep_c_depth,
                        seed=seed,
                        extra={
                            "target_ratio": target_ratio,
                            "ratio": nominal_ratio,
                            "nominal_ratio": nominal_ratio,
                            "ratio_with_side_information": input_dim / (latent_dim + 3),
                        },
                    )
                )
    return rows


def run_sweep_d(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Vary latent dimension at fixed block length."""
    input_dim = config.sweep_d_input_dim
    training, validation, test = _load_autoencoder_data(config, input_dim)
    rows: list[dict[str, Any]] = []
    for latent_dim in config.sweep_d_latent_dims:
        nominal_ratio = input_dim / latent_dim
        for seed in range(config.seeds):
            rows.extend(
                _train_pair(
                    config,
                    training,
                    validation,
                    test,
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    depth_index=config.sweep_d_depth,
                    seed=seed,
                    extra={
                        "ratio": nominal_ratio,
                        "nominal_ratio": nominal_ratio,
                        "ratio_with_side_information": input_dim / (latent_dim + 3),
                    },
                )
            )
    return rows


def evaluate_rice(
    windows: RiceWindows,
    *,
    precision: float = 0.01,
    raw_bits_per_value: int = 32,
) -> list[dict[str, Any]]:
    """Serialize, deserialize, and evaluate each Rice-Golomb test window."""
    rows = []
    for index, raw_differences in enumerate(windows.differences):
        fixed_point = np.round(raw_differences / precision).astype(np.int64)
        mapped = zigzag_encode(fixed_point)
        parameter, theoretical_bits = best_rice_parameter(mapped)
        payload = rice_encode(mapped, parameter)
        decoded = rice_decode(BitReader(payload.bits), len(mapped), parameter)
        reconstructed_differences = (
            zigzag_decode(decoded).astype(np.float32) * precision
        )
        expected = windows.reconstruct(index, raw_differences)
        actual = windows.reconstruct(index, reconstructed_differences)
        rows.append(
            {
                "method": "rice_golomb",
                "window": index,
                "k": parameter,
                "ratio": len(fixed_point) * raw_bits_per_value / len(payload),
                "compressed_bits": len(payload),
                "theoretical_bits": theoretical_bits,
                "mse": float(np.mean((expected - actual) ** 2)),
                "mae": float(np.mean(np.abs(expected - actual))),
            }
        )
    return rows


def evaluate_gorilla(
    windows: np.ndarray,
    *,
    raw_bits_per_value: int = 32,
) -> list[dict[str, Any]]:
    """Serialize, deserialize, and evaluate each Gorilla test window."""
    rows = []
    for index, expected in enumerate(windows):
        payload = gorilla_encode(expected)
        actual = gorilla_decode(BitReader(payload.bits), len(expected))
        rows.append(
            {
                "method": "gorilla",
                "window": index,
                "ratio": len(expected) * raw_bits_per_value / len(payload),
                "compressed_bits": len(payload),
                "mse": float(np.mean((expected - actual) ** 2)),
                "mae": float(np.mean(np.abs(expected - actual))),
            }
        )
    return rows


def run_rice(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_dim in config.baseline_input_dims:
        _, _, test = load_windowed_splits(
            config.dataset,
            input_dim,
            kind="rice",
            training_strides=config.training_strides,
        )
        if not isinstance(test, RiceWindows):
            raise TypeError("Rice slicer returned an unexpected data type")
        window_rows = evaluate_rice(test)
        for row in window_rows:
            row["input_dim"] = input_dim
        rows.extend(window_rows)
    return rows


def run_gorilla(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_dim in config.baseline_input_dims:
        _, _, test = load_windowed_splits(
            config.dataset,
            input_dim,
            kind="gorilla",
            training_strides=config.training_strides,
        )
        if not isinstance(test, np.ndarray):
            raise TypeError("Gorilla slicer returned an unexpected data type")
        window_rows = evaluate_gorilla(test)
        for row in window_rows:
            row["input_dim"] = input_dim
        rows.extend(window_rows)
    return rows


RUNNERS = {
    "b": run_sweep_b,
    "c": run_sweep_c,
    "d": run_sweep_d,
    "rice": run_rice,
    "gorilla": run_gorilla,
}


def save_results(rows: list[dict[str, Any]], destination: Path) -> None:
    """Atomically replace one result table."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(destination)
    print(f"saved {len(rows)} rows to {destination}")


def run_selected(
    config: ExperimentConfig,
    sweeps: list[str],
    *,
    overwrite: bool = False,
) -> None:
    """Run selected experiments and write one CSV per sweep."""
    unknown = set(sweeps) - set(RUNNERS)
    if unknown:
        raise ValueError(f"unknown sweeps: {', '.join(sorted(unknown))}")

    for sweep in sweeps:
        destination = config.results_dir / SWEEP_FILENAMES[sweep]
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        print(f"\n=== Running sweep {sweep} on {config.resolved_device} ===")
        save_results(RUNNERS[sweep](config), destination)

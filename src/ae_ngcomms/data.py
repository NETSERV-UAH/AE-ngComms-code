"""Dataset splitting, windowing, normalization, and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    """Locations of the raw trace and generated chronological splits."""

    raw: Path
    derived_dir: Path

    @property
    def train(self) -> Path:
        return self.derived_dir / "caples_lake_train.csv"

    @property
    def validation(self) -> Path:
        return self.derived_dir / "caples_lake_validation.csv"

    @property
    def test(self) -> Path:
        return self.derived_dir / "caples_lake_test.csv"


@dataclass(frozen=True)
class AutoencoderWindows:
    """Normalized first-difference windows and reconstruction side information."""

    data: np.ndarray
    minimums: np.ndarray
    maximums: np.ndarray
    references: np.ndarray

    def __len__(self) -> int:
        return len(self.data)

    def reconstruct(self, index: int, normalized: np.ndarray) -> np.ndarray:
        differences = (
            normalized * (self.maximums[index] - self.minimums[index])
            + self.minimums[index]
        )
        return np.cumsum(differences) + self.references[index]


@dataclass(frozen=True)
class RiceWindows:
    """Unnormalized first differences and per-window reference values."""

    differences: np.ndarray
    references: np.ndarray

    def __len__(self) -> int:
        return len(self.differences)

    def reconstruct(self, index: int, differences: np.ndarray) -> np.ndarray:
        return np.cumsum(differences) + self.references[index]


def chronological_split(
    frame: pd.DataFrame,
    *,
    validation_size: int = 9_220,
    test_year: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the chronological 79,710/9,220/26,305 split used in the study."""
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")

    # The source subset contains one otherwise valid timestamp prefixed by "#".
    # Normalize that transcription artefact without changing the archived CSV.
    date_text = frame.iloc[:, 0].astype(str).str.lstrip("#")
    dates = pd.to_datetime(date_text, format="mixed", errors="coerce")
    if dates.isna().any():
        invalid = int(dates.isna().sum())
        raise ValueError(f"the date column contains {invalid} unparseable rows")

    test_mask = dates.dt.year >= test_year
    train_validation = frame.loc[~test_mask]
    test = frame.loc[test_mask]
    if validation_size >= len(train_validation):
        raise ValueError("validation split leaves no training observations")

    train = train_validation.iloc[:-validation_size]
    validation = train_validation.iloc[-validation_size:]
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def ensure_chronological_splits(paths: DatasetPaths) -> DatasetPaths:
    """Materialize deterministic split CSVs when they are not already cached."""
    destinations = (paths.train, paths.validation, paths.test)
    if all(path.exists() for path in destinations):
        return paths
    if not paths.raw.exists():
        raise FileNotFoundError(f"dataset not found: {paths.raw}")

    frame = pd.read_csv(paths.raw)
    train, validation, test = chronological_split(frame)
    paths.derived_dir.mkdir(parents=True, exist_ok=True)
    for subset, destination in zip(
        (train, validation, test),
        destinations,
        strict=True,
    ):
        subset.to_csv(destination, index=False)
    return paths


def load_temperatures(csv_path: Path) -> np.ndarray:
    """Load the second CSV column as a contiguous float32 temperature array."""
    values = pd.read_csv(csv_path, usecols=[1], dtype=np.float32).iloc[:, 0]
    return values.to_numpy(dtype=np.float32, copy=True)


def window_starts(
    length: int,
    window_size: int,
    strides: tuple[int, ...] | list[int],
    *,
    shuffle_seed: int | None = None,
) -> list[int]:
    """Return unique one-based starts for every requested stride."""
    if length <= 0 or window_size <= 0:
        raise ValueError("length and window_size must be positive")
    if window_size > length:
        return []

    starts: list[int] = []
    for requested_stride in strides:
        stride = requested_stride if requested_stride > 0 else window_size
        starts.extend(range(1, length - window_size + 2, stride))
    starts = list(dict.fromkeys(starts))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(starts)
    return starts


def _window_with_reference(
    temperatures: np.ndarray,
    start: int,
    window_size: int,
) -> tuple[np.ndarray, float]:
    """Read a block and, where available, the immediately preceding value."""
    zero_based = start - 1
    if zero_based == 0:
        window = temperatures[:window_size]
        return window, float(window[0])
    window = temperatures[zero_based - 1 : zero_based + window_size]
    return window, float(window[0])


def _differences(window: np.ndarray, window_size: int) -> np.ndarray:
    differences = np.diff(window, prepend=window[0])
    return np.nan_to_num(differences, nan=0.0)[-window_size:].astype(
        np.float32,
        copy=False,
    )


def slice_autoencoder_windows(
    csv_path: Path,
    window_size: int,
    strides: tuple[int, ...] | list[int],
    *,
    shuffle_seed: int | None = None,
) -> AutoencoderWindows:
    """Build per-window min-max-normalized first differences."""
    temperatures = load_temperatures(csv_path)
    starts = window_starts(
        len(temperatures),
        window_size,
        strides,
        shuffle_seed=shuffle_seed,
    )
    normalized = np.empty((len(starts), window_size), dtype=np.float32)
    minimums = np.empty(len(starts), dtype=np.float32)
    maximums = np.empty(len(starts), dtype=np.float32)
    references = np.empty(len(starts), dtype=np.float32)

    for index, start in enumerate(starts):
        window, references[index] = _window_with_reference(
            temperatures,
            start,
            window_size,
        )
        differences = _differences(window, window_size)
        minimums[index] = differences.min()
        maximums[index] = differences.max()
        scale = maximums[index] - minimums[index]
        if scale:
            normalized[index] = (differences - minimums[index]) / scale
        else:
            normalized[index].fill(0.0)

    return AutoencoderWindows(normalized, minimums, maximums, references)


def slice_rice_windows(
    csv_path: Path,
    window_size: int,
    strides: tuple[int, ...] | list[int],
) -> RiceWindows:
    """Build unnormalized first-difference windows for Rice-Golomb coding."""
    temperatures = load_temperatures(csv_path)
    starts = window_starts(len(temperatures), window_size, strides)
    differences = np.empty((len(starts), window_size), dtype=np.float32)
    references = np.empty(len(starts), dtype=np.float32)
    for index, start in enumerate(starts):
        window, references[index] = _window_with_reference(
            temperatures,
            start,
            window_size,
        )
        differences[index] = _differences(window, window_size)
    return RiceWindows(differences, references)


def slice_gorilla_windows(
    csv_path: Path,
    window_size: int,
    strides: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Build absolute-temperature windows for Gorilla XOR coding."""
    temperatures = load_temperatures(csv_path)
    starts = window_starts(len(temperatures), window_size, strides)
    windows = np.empty((len(starts), window_size), dtype=np.float32)
    for index, start in enumerate(starts):
        zero_based = start - 1
        windows[index] = temperatures[zero_based : zero_based + window_size]
    return windows


WindowKind = Literal["autoencoder", "rice", "gorilla"]


def load_windowed_splits(
    paths: DatasetPaths,
    window_size: int,
    *,
    kind: WindowKind = "autoencoder",
    training_strides: tuple[int, ...] = (10, 27, 33),
    data_seed: int = 0,
) -> tuple[AutoencoderWindows | RiceWindows | np.ndarray, ...]:
    """Load train, validation, and test windows using the study protocol."""
    ensure_chronological_splits(paths)
    split_paths = (paths.train, paths.validation, paths.test)
    split_strides = (
        training_strides,
        (window_size,),
        (window_size,),
    )

    if kind == "autoencoder":
        return tuple(
            slice_autoencoder_windows(
                path,
                window_size,
                strides,
                shuffle_seed=data_seed if index == 0 else None,
            )
            for index, (path, strides) in enumerate(
                zip(split_paths, split_strides, strict=True)
            )
        )
    if kind == "rice":
        return tuple(
            slice_rice_windows(path, window_size, strides)
            for path, strides in zip(split_paths, split_strides, strict=True)
        )
    if kind == "gorilla":
        return tuple(
            slice_gorilla_windows(path, window_size, strides)
            for path, strides in zip(split_paths, split_strides, strict=True)
        )
    raise ValueError(f"unknown window kind: {kind}")

from pathlib import Path

import numpy as np
import pandas as pd

from ae_ngcomms.data import (
    chronological_split,
    slice_autoencoder_windows,
    window_starts,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_published_split_sizes() -> None:
    frame = pd.read_csv(REPOSITORY / "data/Caples_Lake_N7_2014_2017.csv")
    train, validation, test = chronological_split(frame)
    assert (len(train), len(validation), len(test)) == (79_710, 9_220, 26_305)


def test_window_starts_are_unique_and_bounded() -> None:
    starts = window_starts(20, 5, [2, 3])
    assert starts == list(dict.fromkeys(starts))
    assert starts[0] == 1
    assert all(1 <= start <= 16 for start in starts)


def test_window_reconstruction_recovers_original_values(tmp_path: Path) -> None:
    temperatures = np.array([10.0, 10.5, 11.0, 10.75, 11.25], dtype=np.float32)
    csv_path = tmp_path / "trace.csv"
    pd.DataFrame(
        {
            "date_time": pd.date_range("2020-01-01", periods=len(temperatures)),
            "Temp, C": temperatures,
        }
    ).to_csv(csv_path, index=False)
    windows = slice_autoencoder_windows(csv_path, window_size=3, strides=[2])
    np.testing.assert_allclose(
        windows.reconstruct(0, windows.data[0]),
        temperatures[:3],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        windows.reconstruct(1, windows.data[1]),
        temperatures[2:5],
        atol=1e-6,
    )

import numpy as np

from ae_ngcomms.codecs import (
    BitReader,
    best_rice_parameter,
    gorilla_decode,
    gorilla_encode,
    rice_decode,
    rice_encode,
    zigzag_decode,
    zigzag_encode,
)


def test_rice_round_trip() -> None:
    signed = np.array([0, -1, 1, -2, 7, -100, 2048], dtype=np.int64)
    mapped = zigzag_encode(signed)
    parameter, expected_bits = best_rice_parameter(mapped)
    payload = rice_encode(mapped, parameter)
    decoded = rice_decode(BitReader(payload.bits), len(mapped), parameter)
    np.testing.assert_array_equal(zigzag_decode(decoded), signed)
    assert len(payload) == expected_bits


def test_gorilla_round_trip_is_bit_exact() -> None:
    values = np.array(
        [12.25, 12.25, -0.0, 1.0, 1.0001, 20.5, 20.5],
        dtype=np.float32,
    )
    payload = gorilla_encode(values)
    decoded = gorilla_decode(BitReader(payload.bits), len(values))
    np.testing.assert_array_equal(decoded.view(np.uint32), values.view(np.uint32))

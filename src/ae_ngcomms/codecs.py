"""Bit-level Rice-Golomb and Gorilla baselines used in the article."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FLOAT32_BITS = 32


@dataclass
class BitWriter:
    """Minimal in-memory bit stream used to measure encoded length exactly."""

    bits: list[int] = field(default_factory=list)

    def write_bit(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("a bit must be zero or one")
        self.bits.append(bit)

    def write_bits(self, value: int, width: int) -> None:
        if width < 0:
            raise ValueError("bit width cannot be negative")
        if value < 0 or value >= 1 << width:
            raise ValueError(f"value {value} does not fit in {width} bits")
        for shift in range(width - 1, -1, -1):
            self.bits.append((value >> shift) & 1)

    def __len__(self) -> int:
        return len(self.bits)


@dataclass
class BitReader:
    """Sequential reader for a :class:`BitWriter` payload."""

    bits: list[int]
    position: int = 0

    def read_bit(self) -> int:
        if self.position >= len(self.bits):
            raise EOFError("unexpected end of bit stream")
        bit = self.bits[self.position]
        self.position += 1
        return bit

    def read_bits(self, width: int) -> int:
        if width < 0:
            raise ValueError("bit width cannot be negative")
        value = 0
        for _ in range(width):
            value = (value << 1) | self.read_bit()
        return value


def zigzag_encode(values: np.ndarray) -> np.ndarray:
    """Map signed integers to non-negative integers."""
    integers = np.asarray(values, dtype=np.int64)
    return np.where(integers >= 0, 2 * integers, -2 * integers - 1).astype(np.int64)


def zigzag_decode(values: np.ndarray) -> np.ndarray:
    """Invert :func:`zigzag_encode`."""
    mapped = np.asarray(values, dtype=np.int64)
    return np.where(mapped % 2 == 0, mapped // 2, -(mapped + 1) // 2)


def rice_encoded_length(mapped: np.ndarray, parameter: int) -> int:
    """Return the exact Rice payload length, excluding stream framing."""
    if parameter < 0:
        raise ValueError("Rice parameter cannot be negative")
    values = np.asarray(mapped, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("Rice coding expects non-negative integers")
    quotients = values >> parameter
    return int(np.sum(quotients + 1 + parameter))


def best_rice_parameter(
    mapped: np.ndarray,
    *,
    maximum: int = 20,
) -> tuple[int, int]:
    """Find the Rice parameter with the shortest payload."""
    if maximum < 0:
        raise ValueError("maximum cannot be negative")
    candidates = [
        (parameter, rice_encoded_length(mapped, parameter))
        for parameter in range(maximum + 1)
    ]
    return min(candidates, key=lambda candidate: candidate[1])


def rice_encode(mapped: np.ndarray, parameter: int) -> BitWriter:
    """Encode non-negative integers as unary quotients and binary remainders."""
    values = np.asarray(mapped, dtype=np.int64)
    expected = rice_encoded_length(values, parameter)
    writer = BitWriter()
    remainder_mask = (1 << parameter) - 1
    for value in values:
        integer = int(value)
        quotient = integer >> parameter
        remainder = integer & remainder_mask
        writer.bits.extend([1] * quotient)
        writer.write_bit(0)
        writer.write_bits(remainder, parameter)
    if len(writer) != expected:
        raise AssertionError("Rice length calculation and serialization disagree")
    return writer


def rice_decode(reader: BitReader, count: int, parameter: int) -> np.ndarray:
    """Decode ``count`` values from a Rice payload."""
    if count < 0 or parameter < 0:
        raise ValueError("count and parameter cannot be negative")
    values = np.empty(count, dtype=np.int64)
    for index in range(count):
        quotient = 0
        while reader.read_bit():
            quotient += 1
        remainder = reader.read_bits(parameter)
        values[index] = (quotient << parameter) + remainder
    return values


def gorilla_encode(values: np.ndarray) -> BitWriter:
    """Encode float32 values using Gorilla-style XOR window reuse."""
    float_values = np.asarray(values, dtype=np.float32)
    if float_values.ndim != 1 or not len(float_values):
        raise ValueError("Gorilla coding expects a non-empty one-dimensional array")

    words = float_values.view(np.uint32)
    writer = BitWriter()
    writer.write_bits(int(words[0]), FLOAT32_BITS)
    previous_leading: int | None = None
    previous_trailing: int | None = None
    previous = int(words[0])

    for word in words[1:]:
        current = int(word)
        xor = current ^ previous
        if xor == 0:
            writer.write_bit(0)
            previous = current
            continue

        writer.write_bit(1)
        leading = FLOAT32_BITS - xor.bit_length()
        trailing = (xor & -xor).bit_length() - 1
        meaningful = FLOAT32_BITS - leading - trailing

        can_reuse = (
            previous_leading is not None
            and previous_trailing is not None
            and leading >= previous_leading
            and trailing >= previous_trailing
        )
        if can_reuse:
            writer.write_bit(0)
            width = FLOAT32_BITS - previous_leading - previous_trailing
            payload = (xor >> previous_trailing) & ((1 << width) - 1)
            writer.write_bits(payload, width)
        else:
            writer.write_bit(1)
            payload = (xor >> trailing) & ((1 << meaningful) - 1)
            writer.write_bits(leading, 5)
            writer.write_bits(meaningful - 1, 5)
            writer.write_bits(payload, meaningful)
            previous_leading = leading
            previous_trailing = trailing
        previous = current
    return writer


def gorilla_decode(reader: BitReader, count: int) -> np.ndarray:
    """Decode ``count`` float32 values from a Gorilla payload."""
    if count <= 0:
        raise ValueError("count must be positive")

    reconstructed = np.empty(count, dtype=np.uint32)
    reconstructed[0] = reader.read_bits(FLOAT32_BITS)
    previous_leading: int | None = None
    previous_trailing: int | None = None
    previous = int(reconstructed[0])

    for index in range(1, count):
        if reader.read_bit() == 0:
            current = previous
        elif reader.read_bit() == 0:
            if previous_leading is None or previous_trailing is None:
                raise ValueError("invalid Gorilla stream: no previous XOR window")
            width = FLOAT32_BITS - previous_leading - previous_trailing
            xor = reader.read_bits(width) << previous_trailing
            current = previous ^ xor
        else:
            leading = reader.read_bits(5)
            meaningful = reader.read_bits(5) + 1
            trailing = FLOAT32_BITS - leading - meaningful
            if trailing < 0:
                raise ValueError("invalid Gorilla XOR window")
            xor = reader.read_bits(meaningful) << trailing
            previous_leading = leading
            previous_trailing = trailing
            current = previous ^ xor
        reconstructed[index] = np.uint32(current)
        previous = current
    return reconstructed.view(np.float32)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    val_end: int
    test_end: int


def make_time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> TimeSplit:
    if n <= 0:
        raise ValueError("n must be > 0")

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    test_end = n

    if not (0 < train_end < val_end < test_end):
        raise ValueError("Invalid split fractions for given n")

    return TimeSplit(train_end=train_end, val_end=val_end, test_end=test_end)
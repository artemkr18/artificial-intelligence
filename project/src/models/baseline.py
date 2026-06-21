from __future__ import annotations

import numpy as np


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    last = float(history[-1])
    return np.full(shape=(horizon,), fill_value=last, dtype=float)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    if len(history) < season:
        return naive_forecast(history, horizon)

    last_season = history[-season:]
    reps = int(np.ceil(horizon / season))
    out = np.tile(last_season, reps=reps)[:horizon]
    return out.astype(float)


def moving_average_forecast(history: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    w = min(window, len(history))
    m = float(np.mean(history[-w:]))
    return np.full(shape=(horizon,), fill_value=m, dtype=float)
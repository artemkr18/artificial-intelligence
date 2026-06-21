from __future__ import annotations

import numpy as np


def make_lagged_supervised(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует ряд в supervised-датасет:
    X[t] = [y[t-1], ..., y[t-lags]]
    y[t] = y[t]
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D array")
    if lags <= 0:
        raise ValueError("lags must be > 0")
    if len(series) <= lags:
        raise ValueError("series length must be > lags")

    X = []
    y = []
    for t in range(lags, len(series)):
        X.append(series[t - lags : t][::-1])  # newest first
        y.append(series[t])
    return np.array(X, dtype=float), np.array(y, dtype=float)
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


def recursive_forecast(model: BaseEstimator, history: np.ndarray, horizon: int, lags: int) -> np.ndarray:
    """
    Рекурсивный прогноз на horizon шагов: модель предсказывает 1 шаг,
    затем предсказание добавляется в историю и т.д.
    """
    if len(history) < lags:
        raise ValueError(f"history length must be >= lags ({lags})")

    buf = history.astype(float).tolist()
    preds: list[float] = []

    for _ in range(horizon):
        x = np.array(buf[-lags:][::-1], dtype=float).reshape(1, -1)
        y_next = float(model.predict(x)[0])
        y_next = max(0.0, y_next)  # продажи неотрицательные
        preds.append(y_next)
        buf.append(y_next)

    return np.array(preds, dtype=float)
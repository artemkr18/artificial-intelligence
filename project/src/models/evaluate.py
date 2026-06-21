from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return Metrics(mae=mae, rmse=rmse)


def backtest(
    series: np.ndarray,
    forecast_fn: Callable[[np.ndarray, int], np.ndarray],
    start_idx: int,
    end_idx: int,
    horizon: int,
    stride: int = 7,
) -> Metrics:
    """
    Rolling backtest: в точке t используем history=series[:t],
    предсказываем horizon, сравниваем с series[t:t+horizon].
    """
    y_true_all = []
    y_pred_all = []

    t = start_idx
    while t + horizon <= end_idx:
        history = series[:t]
        y_true = series[t : t + horizon]
        y_pred = forecast_fn(history, horizon)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        t += stride

    if not y_true_all:
        raise ValueError("Backtest produced zero windows")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return compute_metrics(y_true_all, y_pred_all)
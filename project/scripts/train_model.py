from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from src.config import get_settings
from src.data.io import read_sales_csv
from src.data.preprocess import ensure_daily_frequency
from src.data.split import make_time_split
from src.features.lags import make_lagged_supervised
from src.models.baseline import moving_average_forecast, naive_forecast, seasonal_naive_forecast
from src.models.evaluate import backtest
from src.models.forecast import recursive_forecast
from src.models.serialize import save_json, save_model
from src.models.train import train_ridge_model


def main() -> None:
    settings = get_settings()

    raw_path = "data/raw/oil_sales_synth.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Dataset not found: {raw_path}. Run scripts/generate_data.py")

    df = read_sales_csv(raw_path)
    df = ensure_daily_frequency(df, fill_method="ffill")

    series = df["sales"].to_numpy(dtype=float)
    split = make_time_split(len(series), train_frac=0.7, val_frac=0.15)

    horizon = settings.default_horizon
    lags = settings.lags

    # ---- Baselines (val/test) via backtest ----
    baseline_val = {
        "naive": backtest(series, lambda h, H: naive_forecast(h, H), split.train_end, split.val_end, horizon, stride=7),
        "seasonal_naive_7": backtest(
            series, lambda h, H: seasonal_naive_forecast(h, H, season=7), split.train_end, split.val_end, horizon, stride=7
        ),
        "moving_avg_7": backtest(
            series, lambda h, H: moving_average_forecast(h, H, window=7), split.train_end, split.val_end, horizon, stride=7
        ),
    }

    baseline_test = {
        "naive": backtest(series, lambda h, H: naive_forecast(h, H), split.val_end, split.test_end, horizon, stride=7),
        "seasonal_naive_7": backtest(
            series, lambda h, H: seasonal_naive_forecast(h, H, season=7), split.val_end, split.test_end, horizon, stride=7
        ),
        "moving_avg_7": backtest(
            series, lambda h, H: moving_average_forecast(h, H, window=7), split.val_end, split.test_end, horizon, stride=7
        ),
    }

    # ---- Train final model (1-step supervised) ----
    X, y = make_lagged_supervised(series, lags=lags)

    train_end_sup = split.train_end - lags
    if train_end_sup <= 0:
        raise ValueError("Train split too small for given lags")

    X_train, y_train = X[:train_end_sup], y[:train_end_sup]

    model = train_ridge_model(X_train, y_train).model

    # ---- Evaluate final model via multi-step backtest ----
    model_val = backtest(
        series,
        lambda h, H: recursive_forecast(model, np.asarray(h, dtype=float), H, lags=lags),
        split.train_end,
        split.val_end,
        horizon,
        stride=7,
    )
    model_test = backtest(
        series,
        lambda h, H: recursive_forecast(model, np.asarray(h, dtype=float), H, lags=lags),
        split.val_end,
        split.test_end,
        horizon,
        stride=7,
    )

    # ---- Save artifacts ----
    os.makedirs("artifacts", exist_ok=True)
    save_model(model, settings.model_path)

    version = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    meta = {
        "version": version,
        "created_at_utc": datetime.utcnow().isoformat(),
        "frequency": "1D",
        "horizon": horizon,
        "lags": lags,
        "model_type": "sklearn.Pipeline(StandardScaler + Ridge)",
        "data": {
            "source": "synthetic",
            "path": raw_path,
            "rows": int(len(df)),
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max()),
        },
    }
    save_json(meta, settings.meta_path)

    metrics = {
        "val": {
            "baseline": {k: {"mae": v.mae, "rmse": v.rmse} for k, v in baseline_val.items()},
            "model": {"mae": model_val.mae, "rmse": model_val.rmse},
        },
        "test": {
            "baseline": {k: {"mae": v.mae, "rmse": v.rmse} for k, v in baseline_test.items()},
            "model": {"mae": model_test.mae, "rmse": model_test.rmse},
        },
    }
    save_json(metrics, "artifacts/metrics.json")

    print("Training complete.")
    print(json.dumps(meta, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
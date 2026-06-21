from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenConfig:
    seed: int = int(os.getenv("SEED", "42"))
    start_date: str = os.getenv("START_DATE", "2022-01-01")
    days: int = int(os.getenv("DAYS", "1095"))  # 3 years ~ 1095 days
    out_path: str = os.getenv("OUT_PATH", "data/raw/oil_sales_synth.csv")


def generate(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, periods=cfg.days, freq="D")
    t = np.arange(cfg.days, dtype=float)

    # Base level + mild trend
    base = 120.0
    trend = 0.01 * t  # slow growth

    # Seasonality
    weekly = 6.0 * np.sin(2 * np.pi * t / 7.0)       # weekly
    yearly = 10.0 * np.sin(2 * np.pi * t / 365.0)    # yearly

    # Noise
    noise = rng.normal(loc=0.0, scale=3.0, size=cfg.days)

    sales = base + trend + weekly + yearly + noise

    # Rare shocks
    n_shocks = 6
    for _ in range(n_shocks):
        shock_start = int(rng.integers(low=0, high=cfg.days - 10))
        shock_len = int(rng.integers(low=2, high=8))
        shock_mag = float(rng.normal(loc=0.0, scale=15.0))
        sales[shock_start : shock_start + shock_len] += shock_mag

    # Non-negative constraint
    sales = np.clip(sales, 0.0, None)

    df = pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "sales": sales.astype(float),
        }
    )
    return df


def main() -> None:
    cfg = GenConfig()
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)

    df = generate(cfg)
    df.to_csv(cfg.out_path, index=False)

    print(f"Saved: {cfg.out_path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} .. {df['date'].max()}")
    print(
        "Sales stats: "
        f"min={df['sales'].min():.3f}, "
        f"max={df['sales'].max():.3f}, "
        f"mean={df['sales'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
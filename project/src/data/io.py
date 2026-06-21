from __future__ import annotations

import pandas as pd


def read_sales_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("CSV must contain columns: date, sales")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["sales"] = pd.to_numeric(df["sales"], errors="raise").astype(float)
    return df
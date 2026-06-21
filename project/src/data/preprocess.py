from __future__ import annotations

import pandas as pd


def ensure_daily_frequency(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Гарантирует непрерывную дневную сетку дат (freq='D').
    Если есть пропуски дат — заполняет sales выбранным методом.
    """
    out = df.copy()
    out = out.set_index("date").sort_index()

    full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full_index)
    out.index.name = "date"

    if fill_method == "ffill":
        out["sales"] = out["sales"].ffill()
        out["sales"] = out["sales"].fillna(0.0)
    elif fill_method == "interpolate":
        out["sales"] = out["sales"].interpolate(limit_direction="both")
        out["sales"] = out["sales"].fillna(0.0)
    elif fill_method == "zero":
        out["sales"] = out["sales"].fillna(0.0)
    else:
        raise ValueError("fill_method must be one of: ffill, interpolate, zero")

    out = out.reset_index()
    out["sales"] = out["sales"].astype(float)
    return out
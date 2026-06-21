from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    history: List[float] = Field(..., description="Historical daily sales values (float).")
    horizon: Optional[int] = Field(None, description="Forecast horizon in days. Default=30.")

    @field_validator("history")
    @classmethod
    def validate_history(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("history must be non-empty")
        for x in v:
            if x != x:
                raise ValueError("history contains NaN")
            if x in (float("inf"), float("-inf")):
                raise ValueError("history contains inf")
        return v


class PredictResponse(BaseModel):
    horizon: int
    forecast: List[float]
    model_version: str
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException

from src.config import get_settings
from src.models.forecast import recursive_forecast
from src.service.deps import STATE, is_model_loaded, load_artifacts, model_version
from src.service.schemas import PredictRequest, PredictResponse
from src.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("service")
    logger.info("Starting app (env=%s)", settings.app_env)

    load_artifacts(settings.model_path, settings.meta_path)
    yield

    logger.info("Shutting down app")


app = FastAPI(
    title="Oil Sales Forecasting Service",
    version="0.2.0",
    lifespan=lifespan,
)

logger = logging.getLogger("service")


@app.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_version": model_version(),
        "app_env": settings.app_env,
        "error": STATE.error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    settings = get_settings()

    if not is_model_loaded():
        raise HTTPException(status_code=500, detail="Model is not loaded")

    history = np.asarray(req.history, dtype=float)

    if len(history) < settings.min_history:
        raise HTTPException(
            status_code=400,
            detail=f"history length must be >= {settings.min_history}",
        )

    horizon = req.horizon if req.horizon is not None else settings.default_horizon
    if horizon <= 0 or horizon > settings.default_horizon:
        raise HTTPException(
            status_code=400,
            detail=f"horizon must be in range 1..{settings.default_horizon}",
        )

    t0 = time.time()
    forecast = recursive_forecast(
        model=STATE.model,
        history=history,
        horizon=horizon,
        lags=settings.lags,
    )
    dt_ms = (time.time() - t0) * 1000.0

    logger.info(
        "predict ok | history_len=%d horizon=%d latency_ms=%.2f",
        len(history),
        horizon,
        dt_ms,
    )

    return PredictResponse(
        horizon=horizon,
        forecast=[float(x) for x in forecast.tolist()],
        model_version=model_version(),
    )
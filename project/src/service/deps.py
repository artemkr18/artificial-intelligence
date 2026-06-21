from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import joblib

logger = logging.getLogger("service.deps")


@dataclass
class ModelState:
    model: Optional[Any] = None
    meta: Optional[dict] = None
    error: Optional[str] = None


STATE = ModelState()


def load_artifacts(model_path: str, meta_path: str) -> None:
    STATE.model = None
    STATE.meta = None
    STATE.error = None

    if not os.path.exists(model_path):
        STATE.error = f"Model file not found: {model_path}"
        logger.error(STATE.error)
        return

    try:
        STATE.model = joblib.load(model_path)
    except Exception as e:
        STATE.error = f"Failed to load model: {e}"
        logger.exception("Failed to load model")
        return

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                STATE.meta = json.load(f)
        except Exception as e:
            logger.warning("Failed to read meta.json: %s", e)
            STATE.meta = None

    logger.info("Model loaded successfully from %s", model_path)


def is_model_loaded() -> bool:
    return STATE.model is not None


def model_version() -> str:
    if not STATE.meta:
        return "none"
    return str(STATE.meta.get("version", "none"))
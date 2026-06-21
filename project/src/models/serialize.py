from __future__ import annotations

import json
import os
from typing import Any, Dict

import joblib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_model(model: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
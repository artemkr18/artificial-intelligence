import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    model_path: str = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    meta_path: str = os.getenv("META_PATH", "artifacts/meta.json")

    min_history: int = int(os.getenv("MIN_HISTORY", "60"))
    default_horizon: int = int(os.getenv("DEFAULT_HORIZON", "30"))
    lags: int = int(os.getenv("LAGS", "60"))


def get_settings() -> Settings:
    return Settings()
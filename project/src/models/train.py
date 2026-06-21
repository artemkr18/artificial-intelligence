from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline


def train_ridge_model(X_train: np.ndarray, y_train: np.ndarray) -> TrainResult:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return TrainResult(model=model)
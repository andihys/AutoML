from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import numpy as np
from typing import Tuple, Dict, Any

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray
) -> BaseEstimator:
    model.fit(X, y)
    return model

def evaluate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    task: str
) -> Dict[str, Any]:
    y_pred = model.predict(X)
    
    if task == "classification":
        score = accuracy_score(y, y_pred)
        metrics = {
            "metric": "accuracy",
            "accuracy": round(score, 4)
        }

    elif task == "regression":
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        metrics = {
            "metric": "r2_score",
            "r2_score": round(r2, 4),
            "mse": round(mse, 4),
            "mae": round(mae, 4)
        }

    else:
        raise ValueError(f"Unsupported task: {task}")

    return metrics

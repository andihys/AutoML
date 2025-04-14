from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
from typing import Tuple

def train_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray
) -> BaseEstimator:
    """
    Trains the given model on the provided data.

    Args:
        model (BaseEstimator): scikit-learn compatible model
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector

    Returns:
        Trained model
    """
    model.fit(X, y)
    return model

def evaluate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Evaluates the trained model using accuracy.

    Args:
        model (BaseEstimator): Trained model
        X (np.ndarray): Feature matrix
        y (np.ndarray): True labels

    Returns:
        Accuracy score (float)
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc

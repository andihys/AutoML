from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import numpy as np
from typing import Tuple

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion of test set
        random_state (int): Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
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
    y: np.ndarray
) -> float:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc

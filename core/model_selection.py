from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from typing import Any

def get_model(name: str, task: str) -> Any:
    """
    Factory function to return a model instance based on its name and task.

    Args:
        name (str): Name of the model.
        task (str): "classification" or "regression"

    Returns:
        scikit-learn model instance
    """
    name = name.lower()
    task = task.lower()

    if task == "classification":
        if name == "randomforest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif name == "logreg":
            return LogisticRegression(max_iter=500)
        elif name == "svm":
            return SVC()
        elif name == "decisiontree":
            return DecisionTreeClassifier()
        else:
            raise ValueError(f"Unsupported classification model: {name}")

    elif task == "regression":
        if name == "randomforest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif name == "linreg":
            return LinearRegression()
        elif name == "svm":
            return SVR()
        elif name == "decisiontree":
            return DecisionTreeRegressor()
        else:
            raise ValueError(f"Unsupported regression model: {name}")

    else:
        raise ValueError(f"Unknown task: {task}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from typing import Any


def get_model(name: str) -> Any:
    """
    Factory function to return a model instance based on its name.

    Args:
        name (str): Name of the model. Supported: "randomforest", "logreg", "svm", "decisiontree"

    Returns:
        scikit-learn model instance
    """
    name = name.lower()

    if name == "randomforest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == "logreg":
        return LogisticRegression(max_iter=500)
    elif name == "svm":
        return SVC()
    elif name == "decisiontree":
        return DecisionTreeClassifier()
    else:
        raise ValueError(f"Unsupported model: {name}")

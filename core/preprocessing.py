import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple

def detect_task(y: pd.Series) -> str:
    """
    Detects whether the task is classification or regression based on the target variable.

    Args:
        y (pd.Series): Target column.

    Returns:
        "classification" or "regression"
    """
    if y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 20:
        return "classification"
    else:
        return "regression"

def preprocess_dataframe(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Preprocess the input DataFrame:
    - Imputes missing values
    - Encodes categoricals
    - Scales numericals
    - Detects ML task type

    Args:
        df (pd.DataFrame): Full input dataset.
        target_col (str): Name of the target column.

    Returns:
        Tuple[X_processed, y, task]
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    task = detect_task(y)

    return X_processed, y, task

import logging
import pandas as pd

from core.preprocessing import preprocess_dataframe
from core.model_selection import get_model
from core.trainer import train_model, evaluate_model

from cli.config_loader import load_config


def run_pipeline(df: pd.DataFrame) -> None:
    config = load_config()
    logging.info("⚙️  Loaded config from config/default.yaml")

    target_col = config["target"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X_processed, y = preprocess_dataframe(df, target_col)
    logging.info(f"✅ Preprocessing completed. Feature matrix shape: {X_processed.shape}")

    model_name = config["model"]["name"]
    model = get_model(model_name)
    logging.info(f"📦 Model selected from config: {model.__class__.__name__}")

    model = train_model(model, X_processed, y)
    logging.info("✅ Model training completed.")

    accuracy = evaluate_model(model, X_processed, y)
    logging.info(f"📊 Accuracy on full dataset: {accuracy:.4f}")

    logging.info("🎉 Pipeline completed.")

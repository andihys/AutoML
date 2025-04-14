import logging
import pandas as pd

from core.preprocessing import preprocess_dataframe
from core.model_selection import get_model
from core.trainer import train_model, evaluate_model

def run_pipeline(df: pd.DataFrame) -> None:
    logging.info("🔧 Starting pipeline...")
    logging.info(f"Dataset shape: {df.shape}")

    target_col = df.columns[-1]
    logging.info(f"🧬 Target column detected: '{target_col}'")

    X_processed, y = preprocess_dataframe(df, target_col)
    logging.info(f"✅ Preprocessing completed. Feature matrix shape: {X_processed.shape}")

    model = get_model("randomforest")
    logging.info(f"📦 Model selected: {model.__class__.__name__}")

    model = train_model(model, X_processed, y)
    logging.info("✅ Model training completed.")

    accuracy = evaluate_model(model, X_processed, y)
    logging.info(f"📊 Accuracy on full dataset: {accuracy:.4f}")

    logging.info("🎉 Pipeline completed.")

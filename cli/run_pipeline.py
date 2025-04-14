import logging
import pandas as pd

from core.preprocessing import preprocess_dataframe
from core.model_selection import get_model

def run_pipeline(df: pd.DataFrame) -> None:
    """
    Orchestrates the AutoML pipeline steps:
    - Preprocessing
    - Model selection
    - (future) Training
    - (future) Evaluation

    Args:
        df (pd.DataFrame): Input dataset including target column
    """
    logging.info("🔧 Starting pipeline...")
    logging.info(f"Dataset shape: {df.shape}")

    # Assume the last column is the target for now
    target_col = df.columns[-1]
    logging.info(f"🧬 Target column detected: '{target_col}'")

    # Step 1: Preprocessing
    X_processed, y = preprocess_dataframe(df, target_col)
    logging.info(f"✅ Preprocessing completed. Feature matrix shape: {X_processed.shape}")

    # Step 2: Model selection
    model = get_model("randomforest")  # Fixed for now, will be configurable later
    logging.info(f"📦 Model selected: {model.__class__.__name__}")

    # Placeholder: Training
    logging.info("🚧 Training step not implemented yet.")

    # Placeholder: Evaluation
    logging.info("🚧 Evaluation step not implemented yet.")

    logging.info("🎉 Pipeline completed.")

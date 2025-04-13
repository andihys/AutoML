import logging
import pandas as pd
from core.preprocessing import preprocess_dataframe

def run_pipeline(df: pd.DataFrame) -> None:
    logging.info("🔧 Starting pipeline...")
    logging.info(f"Dataset shape: {df.shape}")

    target_col = df.columns[-1]  # Assumiamo che l’ultima colonna sia il target
    X_processed, y = preprocess_dataframe(df, target_col)

    logging.info(f"✅ Preprocessing completed. Feature matrix shape: {X_processed.shape}")
    logging.info("🚧 Placeholder: model selection")
    logging.info("🚧 Placeholder: training")
    logging.info("🚧 Placeholder: evaluation")

    logging.info("🎉 Pipeline completed.")

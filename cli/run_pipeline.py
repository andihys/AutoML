import logging
import pandas as pd

from cli.config_loader import load_config
from core.preprocessing import preprocess_dataframe
from core.model_selection import get_model
from core.trainer import split_data, train_model, evaluate_model
from core.saver import save_model
from core.result_saver import save_results

def run_pipeline(df: pd.DataFrame) -> None:
    """
    Orchestrates the AutoML pipeline steps:
    - Load config
    - Preprocessing
    - Model selection
    - Train/test split
    - Training
    - Evaluation
    - Model and results saving

    Args:
        df (pd.DataFrame): Input dataset including target column
    """
    config = load_config()
    logging.info("⚙️  Loaded config from config/default.yaml")

    target_col = config["target"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Step 1: Preprocessing
    X_processed, y = preprocess_dataframe(df, target_col)
    logging.info(f"✅ Preprocessing completed. Feature matrix shape: {X_processed.shape}")

    # Step 2: Model selection
    model_name = config["model"]["name"]
    model = get_model(model_name)
    logging.info(f"📦 Model selected from config: {model.__class__.__name__}")

    # Step 3: Train/Test split
    test_size = config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = split_data(X_processed, y, test_size=test_size)
    logging.info(f"🔀 Data split: {X_train.shape[0]} train / {X_test.shape[0]} test")

    # Step 4: Training
    model = train_model(model, X_train, y_train)
    logging.info("✅ Model training completed.")

    # Step 5: Evaluation
    accuracy = evaluate_model(model, X_test, y_test)
    logging.info(f"📊 Accuracy on test set: {accuracy:.4f}")

    # Step 6: Model saving
    model_path = save_model(model)
    logging.info(f"💾 Model saved at: {model_path}")

    # Step 7: Results saving
    results = {
        "model_name": model.__class__.__name__,
        "accuracy": round(accuracy, 4),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "model_path": model_path
    }
    results_path = save_results(results)
    logging.info(f"📝 Results saved at: {results_path}")

    logging.info("🎉 Pipeline completed.")

import argparse
import logging
import os
import pandas as pd
from cli.run_pipeline import run_pipeline  # <--- Import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoML Micro Framework — CLI Entry Point"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input CSV dataset"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        df = load_dataset(args.dataset)
        run_pipeline(df)  # <--- Use the new orchestrator
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    main()

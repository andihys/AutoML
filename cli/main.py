import typer
import pandas as pd
import logging
import os

from cli.run_pipeline import run_pipeline
from cli.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()

@app.command()
def main(
    dataset: str = typer.Option(..., help="Path to the CSV dataset"),
    config_path: str = typer.Option("config/default.yaml", help="Path to configuration YAML"),
    model_name: str = typer.Option(None, help="Override model name from config"),
    output_dir: str = typer.Option("artifacts", help="Directory to save artifacts")
):
    """
    AutoML Micro Framework - Run a complete ML pipeline from CLI.
    """

    if not os.path.exists(dataset):
        logging.error(f"❌ Dataset not found: {dataset}")
        raise typer.Exit(code=1)

    df = pd.read_csv(dataset)
    logging.info(f"✅ Dataset loaded successfully with shape {df.shape}")

    config = load_config(config_path)

    # Override model name if provided
    if model_name:
        config["model"]["name"] = model_name

    run_pipeline(df, config, output_dir)

if __name__ == "__main__":
    app()

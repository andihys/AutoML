# AutoML Micro Framework 🚀

A lightweight and modular AutoML framework for structured tabular datasets.  
This project allows users to run a complete ML pipeline (preprocessing, model selection, training, evaluation) with minimal configuration.

## ✅ Features

- Modern CLI based on [Typer](https://typer.tiangolo.com/)
- Automatic preprocessing (scaling, encoding, imputing)
- Dynamic model selection via config or CLI
- Train/Test splitting for realistic evaluation
- Model saving (`trained_model.pkl`)
- Metrics saving (`results.json`) for experiment tracking
- Modular structure, ready for extension (tuning, regression, API deployment)

## 🔧 Quickstart

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automl_micro.git
cd automl_micro
````

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Prepare your CSV dataset.
   Ensure the target variable is correctly labeled and referenced inside `config/default.yaml`.

4. Run the pipeline via CLI:

```bash
python cli/main.py --dataset path/to/your.csv
```

Optional arguments:

* `--model_name`: override the model specified in the config
* `--config_path`: use a different YAML configuration
* `--output_dir`: customize the output directory for artifacts

Example:

```bash
python cli/main.py --dataset data/iris.csv --model_name logreg --output_dir my_results/
```

After running:

* The trained model will be saved inside the specified artifacts folder.
* The results (accuracy, model name, train/test sizes) will be appended to `results.json`.

## 📁 Project Structure

```
automl_micro/
├── core/
│   ├── preprocessing.py     # Preprocessing pipeline
│   ├── model_selection.py    # Model selection logic
│   ├── trainer.py             # Training and evaluation
│   ├── saver.py               # Model saving
│   └── result_saver.py        # Saving run metrics
├── cli/
│   ├── main.py                # CLI entry point (Typer)
│   ├── run_pipeline.py        # Pipeline orchestration
│   └── config_loader.py       # Configuration loader
├── config/
│   └── default.yaml           # Default pipeline configuration
├── artifacts/                 # (Created at runtime) Saved models and results
├── requirements.txt
└── README.md
```

## 📌 Configuration

Edit `config/default.yaml` to set:

* Target column
* Preprocessing strategies (scaling, encoding, imputation)
* Model name and hyperparameters
* Test set split ratio

Example:

```yaml
target: species

preprocessing:
  scaling: standard
  encoding: onehot
  imputation_num: mean
  imputation_cat: most_frequent

model:
  name: randomforest
  params:
    n_estimators: 100
    max_depth: 5

test_size: 0.2
```

## 🧪 Coming soon

* Regression task support
* Hyperparameter tuning with Optuna
* Advanced preprocessing (outlier handling, feature engineering)
* Inference API (FastAPI endpoint)
* Experiment tracking dashboard

## 📌 License

MIT License.

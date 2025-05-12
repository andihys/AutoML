# AutoML Micro Framework 🚀

A lightweight and modular AutoML framework for structured tabular datasets.  
This project allows users to run a full ML pipeline (preprocessing, model selection, training, evaluation) with minimal configuration.

## ✅ Features

- Easy-to-use CLI for quick runs
- Automatic preprocessing (scaling, encoding, imputing)
- Dynamic model selection based on config
- Train/Test splitting for realistic evaluation
- Model saving (`trained_model.pkl`)
- Metrics saving (`results.json`) for experiment tracking
- Modular structure, ready for extension (tuning, deployment)

## 🔧 Quickstart

1. Clone the repo:
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
   Make sure your target variable is present and configure it inside `config/default.yaml`.

4. Run the CLI:

```bash
python cli/main.py --dataset path/to/your.csv
```

After running:

* The trained model will be saved inside the `artifacts/` directory.
* The results (accuracy, model name, etc.) will be logged into `artifacts/results.json`.

## 📁 Project Structure

```
automl_micro/
├── core/
│   ├── preprocessing.py   # Preprocessing pipeline
│   ├── model_selection.py # Model selection logic
│   ├── trainer.py          # Training and evaluation
│   ├── saver.py            # Model saving
│   └── result_saver.py     # Saving run metrics
├── cli/
│   ├── main.py             # CLI entry point
│   ├── run_pipeline.py     # Pipeline orchestration
│   └── config_loader.py    # Configuration loader
├── config/
│   └── default.yaml        # Default pipeline configuration
├── artifacts/              # (Created at runtime) Saved models and results
├── requirements.txt
└── README.md
```

## 📌 Configuration

Edit `config/default.yaml` to set:

* Target column name
* Preprocessing strategies (scaling, encoding, imputation)
* Model type and parameters
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

* Hyperparameter tuning
* Inference API (FastAPI)
* More models
* Experiment tracking dashboard

## 📌 License

MIT License.

# AutoML Micro Framework 🚀

A lightweight and modular AutoML framework for structured tabular datasets.  
This project allows users to run a full ML pipeline (preprocessing, model selection, training, evaluation) with minimal configuration.

## ✅ Features (WIP)

- Easy-to-use CLI for quick runs
- Automatic preprocessing (scaling, encoding, imputing)
- Model selection and tuning
- Metrics and evaluation reports
- Ready for experimentation and extension

## 🔧 Quickstart

1. Clone the repo:
```bash
git clone https://github.com/yourusername/automl_micro.git
cd automl_micro
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the CLI on a CSV file:
```bash
python cli/main.py --dataset path/to/your.csv
```

## 📁 Project Structure

- `core/`: main logic (preprocessing, training, etc.)
- `cli/`: command line interface
- `examples/`: example notebooks and scripts
- `config/`: configuration files
- `tests/`: unit tests

## 📌 License

MIT License.

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

## 🚀 What Does It Automate?

1. **Dataset Loading**  
   Accepts standard `.csv` files, even unprocessed ones.

2. **Automatic Preprocessing**  
   - Missing value imputation
   - Categorical variable encoding
   - Feature scaling

3. **Model Selection**  
   - Classification models: RandomForest, XGBoost, LogisticRegression  
   - (Regression support planned)

4. **Hyperparameter Optimization**  
   - Grid Search and/or Optuna integration

5. **Model Training with Cross-Validation**

6. **Evaluation and Reporting**  
   - Accuracy, Precision, Recall, F1, Confusion Matrix

7. **Outputs**  
   - Trained model (`.pkl`)
   - Evaluation metrics (`.json` or `.txt`)
   - Optional: CSV with predictions

---

## 👥 Who Is It For?

- **Data Scientists** needing fast prototyping pipelines
- **ML Engineers** who want to modularize standard workflows
- **Students** learning ML without rebuilding pipelines every time
- **Startups or solo developers** looking for simple but effective AutoML tools

---

## 💡 Example Usage

```bash
python cli/main.py --dataset data/titanic.csv
```

**Outputs:**
- Saved model: `outputs/best_model.pkl`
- Report file: `outputs/report.json`

## 📌 License

MIT License.

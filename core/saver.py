import joblib
import os

def save_model(model, output_dir: str = "artifacts", filename: str = "trained_model.pkl") -> str:
    """
    Saves the trained model to a file.

    Args:
        model: Trained model instance
        output_dir (str): Directory where to save the model
        filename (str): Filename for the saved model

    Returns:
        Full path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    return path

import json
import os
from datetime import datetime

def save_results(results: dict, output_dir: str = "artifacts", filename: str = "results.json") -> str:
    """
    Saves the results dictionary to a JSON file.

    Args:
        results (dict): Dictionary containing run information
        output_dir (str): Directory where to save the results
        filename (str): JSON file name

    Returns:
        Full path to the saved results file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # If file exists, append to it; otherwise create a new list
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    # Add timestamp
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing.append(results)

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)

    return path

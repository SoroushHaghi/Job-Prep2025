# src/plot_comparison.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_comparison(rf_results_path: str, cnn_results_path: str, output_path: str):
    """
    Loads results from two models, creates a comparative bar chart of their
    F1-scores, and saves it to a file.
    """
    try:
        with open(rf_results_path, "r") as f:
            rf_results = json.load(f)
        with open(cnn_results_path, "r") as f:
            cnn_results = json.load(f)
    except FileNotFoundError as e:
        print(
            f"Error: Results file not found. Please run training for both models first. Missing: {e.filename}"
        )
        return

    labels = ["throwing", "drinking", "driving"]
    rf_f1 = [rf_results[label]["f1-score"] for label in labels]
    cnn_f1 = [cnn_results[label]["f1-score"] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, rf_f1, width, label="RandomForest")
    rects2 = ax.bar(x + width / 2, cnn_f1, width, label="1D-CNN")

    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score by Class and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")
    plt.close()

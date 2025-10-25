# src/model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,  # Import confusion matrix
    ConfusionMatrixDisplay,  # Import display function
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt  # Import matplotlib for plotting

import json

# --- Configuration ---
FEATURE_FILE = Path("data/features.csv")
MODEL_FILE = Path("models/activity_classifier.joblib")
CONFUSION_MATRIX_FILE = Path("docs/confusion_matrix.png")  # Path to save the plot
RESULTS_FILE = Path("models/rf_results.json")

# Ensure the directories exist
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)

# Define human-readable labels for the classes (MUST match dataset_builder)
CLASS_NAMES = ["throwing", "drinking", "driving"]
# ---


def train_model():
    """
    Loads features, trains a RandomForestClassifier, evaluates it,
    saves the model, and saves a confusion matrix plot.
    """
    print("Starting multi-class model training process...")

    # 1. Load Data
    try:
        df = pd.read_csv(FEATURE_FILE)
        print(f"Feature data loaded successfully from {FEATURE_FILE}.")
        print(f"Data shape: {df.shape}")
        print("Class distribution in loaded data:")
        # Map labels to names for better readability
        label_map = {i: name for i, name in enumerate(CLASS_NAMES)}
        print(df["label"].map(label_map).value_counts())

    except FileNotFoundError:
        print(f"Error: Feature file not found at {FEATURE_FILE}")
        print("Please run the 'build-features' command first.")
        return  # Exit if data is missing
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Separate features (X) and labels (y)
    X = df.drop("label", axis=1)
    y = df["label"]

    # --- Data Leakage Check (remains the same) ---
    print("\n--- DATA LEAKAGE CHECK ---")
    if len(X) < 2:
        print("Not enough data to perform split and leakage check.")
        return

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Convert back to DataFrames to use index for checking
    X_train_df = pd.DataFrame(X_train_full, index=X_train_full.index)
    X_test_df = pd.DataFrame(X_test_full, index=X_test_full.index)

    overlap = X_train_df.index.intersection(X_test_df.index)
    print(f"Training samples: {len(X_train_df)}")
    print(f"Test samples: {len(X_test_df)}")
    print(f"Overlap (leaked samples): {len(overlap)}")
    if len(overlap) == 0:
        print("--- CHECK PASSED: No data leakage detected via index. ---")
    else:
        print("--- WARNING: Data leakage detected via index! ---")
    # --- End Leakage Check ---

    # 2. Split Data (using the same split as the check)
    X_train, X_test, y_train, y_test = (
        X_train_full,
        X_test_full,
        y_train_full,
        y_test_full,
    )
    print(
        f"\nData split: {len(X_train)} training samples, {len(X_test)} testing samples."
    )
    print("Class distribution in Training set:")
    print(y_train.map(label_map).value_counts())
    print("Class distribution in Test set:")
    print(y_test.map(label_map).value_counts())

    # 3. Train Model (RandomForest handles multi-class automatically)
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # Save classification report
    with open(RESULTS_FILE, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to {RESULTS_FILE}")

    # --- Generate and Save Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    try:
        cm = confusion_matrix(y_test, y_pred)
        # Use class names for display labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

        # Plot and save the figure
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")  # Plot on the axes
        plt.title("Confusion Matrix")
        plt.tight_layout()  # Adjust layout
        plt.savefig(CONFUSION_MATRIX_FILE)
        print(f"Confusion matrix plot saved to {CONFUSION_MATRIX_FILE}")
        plt.close(fig)  # Close the figure to free memory

    except Exception as e:
        print(f"Error generating or saving confusion matrix: {e}")
    # ---

    # 5. Generate and Save Feature Importance Plot
    print("\nGenerating Feature Importance Plot...")
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns

        # Create a pandas series for easy plotting
        feature_importance_series = pd.Series(importances, index=feature_names)

        # Plotting
        plt.figure(figsize=(10, 6))
        feature_importance_series.sort_values(ascending=True).plot(kind="barh")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save the plot
        feature_importance_plot_file = Path("docs/feature_importance.png")
        plt.savefig(feature_importance_plot_file)
        print(f"Feature importance plot saved to {feature_importance_plot_file}")
        plt.close()

    except Exception as e:
        print(f"Error generating or saving feature importance plot: {e}")

    # 6. Save Model
    print(f"\nSaving model to {MODEL_FILE}...")
    try:
        joblib.dump(model, MODEL_FILE)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nModel training script finished.")

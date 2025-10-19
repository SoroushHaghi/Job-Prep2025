# src/model_trainer.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
FEATURES_PATH = os.path.join("data", "features.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "activity_classifier.joblib")


def train_model():
    """
    Loads features, splits data, trains a model, evaluates, and saves it.
    """
    print("Starting model training process...")

    # 1. Load Data
    try:
        dataset = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {FEATURES_PATH}")
        print("Please run the 'dataset_builder.py' script first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Data loaded successfully.")

    # 2. Split Data into Features (X) and Labels (y)
    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    # 3. Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- CHECK THE DATA LEAKAGE ---
    print("\n--- DATA LEAKAGE CHECK ---")
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    overlap = train_indices.intersection(test_indices)

    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Overlap (leaked samples): {len(overlap)}")

    if len(overlap) > 0:
        print("!!! WARNING: DATA LEAKAGE DETECTED !!!")
    else:
        print("--- CHECK PASSED: No data leakage. ---")
    # --- END OF CHECK ---

    print(
        f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples."
    )

    # 4. Initialize and Train the Model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Save the Model
    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")

    # 6. Evaluate the Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")

    # --- UPDATED: Added digits=4 ---
    # This will show the true, non-rounded precision/recall
    # and resolve the 1.00 vs 99.80 confusion.
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stillness (0)", "Movement (1)"],
            digits=4,  # <-- This is the added parameter
        )
    )


if __name__ == "__main__":
    # This allows you to run the script directly from the terminal
    # using: python src/model_trainer.py
    train_model()

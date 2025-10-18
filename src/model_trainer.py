# src/model_trainer.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
FEATURES_PATH = os.path.join("data", "features.csv")


def train_model():
    """
    Loads features, splits data, trains a model, and evaluates it.
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
    # 'label' is our target, everything else is a feature
    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    # 3. Split Data into Training and Testing sets
    # test_size=0.2 means 20% of data is for testing, 80% for training
    # random_state=42 ensures we get the same split every time (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples."
    )

    # 4. Initialize and Train the Model
    print("Training RandomForestClassifier...")
    # n_estimators=100 means it uses 100 "decision trees"
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # The .fit() method is where the model "learns" from the data
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the Model
    print("\n--- Model Evaluation ---")

    # Make predictions on the *unseen* test data
    y_pred = model.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Show detailed report (precision, recall, f1-score)
    print("\nClassification Report:")
    # target_names=['Stillness (0)', 'Movement (1)'] makes the report readable
    print(
        classification_report(
            y_test, y_pred, target_names=["Stillness (0)", "Movement (1)"]
        )
    )


if __name__ == "__main__":
    # This allows you to run the script directly from the terminal
    # using: python src/model_trainer.py
    train_model()

# src/predictor.py
import joblib
import pandas as pd
from pathlib import Path
from . import features  # Import feature extraction functions

# Define the class names (MUST match training order: throwing=0, drinking=1, driving=2)
CLASS_NAMES = ["throwing", "drinking", "driving"]


class ActivityPredictor:
    """
    Handles loading the multi-class model and making predictions.
    """

    def __init__(self, model_path: Path):
        """Loads the trained model."""
        try:
            self.model = joblib.load(model_path)
            # Optional: Keep this confirmation print if you like
            # print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            print("Please run the 'train' command first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Feature order must match training
        self.feature_order = [
            "X_mean",
            "X_std",
            "X_rms",
            "Y_mean",
            "Y_std",
            "Y_rms",
            "Z_mean",
            "Z_std",
            "Z_rms",
        ]

    def predict(self, data_window: list[dict]) -> str:
        """
        Predicts the activity for a window and returns the class name.
        """  # <-- Ensure docstring is closed
        window_df = pd.DataFrame(data_window).astype(float)

        feature_dict = {}
        for axis in ["X", "Y", "Z"]:
            feature_dict[f"{axis}_mean"] = features.calculate_mean(window_df[axis])
            feature_dict[f"{axis}_std"] = features.calculate_std_dev(window_df[axis])
            feature_dict[f"{axis}_rms"] = features.calculate_rms(window_df[axis])

        features_df = pd.DataFrame([feature_dict], columns=self.feature_order)

        # Predict the class index
        prediction_index = self.model.predict(features_df)[0]

        # --- DEBUG LINES REMOVED ---
        # print(f"DEBUG: Raw prediction index: {prediction_index} (Type: {type(prediction_index)})")
        # --- END DEBUG LINES ---

        prediction_name = "Unknown"  # Default in case of error

        # Map the index to the human-readable class name
        try:
            # Explicitly cast to int just in case it's a float like 0.0
            index = int(prediction_index)
            # Check if index is within the valid range for CLASS_NAMES
            if 0 <= index < len(CLASS_NAMES):
                prediction_name = CLASS_NAMES[index]
                # --- DEBUG LINE REMOVED ---
                # print(f"DEBUG: Mapped index {index} to name '{prediction_name}' successfully.")
                # --- END DEBUG LINES ---
            else:
                # Index is out of expected range (0, 1, 2)
                print(
                    f"Warning: Model predicted an out-of-bounds index: {prediction_index}"
                )
                # prediction_name remains "Unknown"

        except (
            TypeError,
            ValueError,
        ) as e:  # Catch potential errors during int conversion
            print(
                f"Warning: Could not convert prediction index {prediction_index} to int. Error: {type(e).__name__} - {e}"
            )
            # prediction_name remains "Unknown"

        # --- DEBUG LINE REMOVED ---
        # print(f"DEBUG: Returning prediction_name: '{prediction_name}'")
        # --- END DEBUG LINES ---

        return prediction_name

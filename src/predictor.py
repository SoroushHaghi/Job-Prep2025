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
        """Loads the trained model."""  # <-- Check this docstring closure
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")  # Confirmation
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
        Includes extra debugging.
        """
        window_df = pd.DataFrame(data_window).astype(float)

        feature_dict = {}
        for axis in ["X", "Y", "Z"]:
            feature_dict[f"{axis}_mean"] = features.calculate_mean(window_df[axis])
            feature_dict[f"{axis}_std"] = features.calculate_std_dev(window_df[axis])
            feature_dict[f"{axis}_rms"] = features.calculate_rms(window_df[axis])

        features_df = pd.DataFrame([feature_dict], columns=self.feature_order)

        # Predict the class index
        prediction_index = self.model.predict(features_df)[0]

        # --- MORE DEBUGGING ---
        print(
            f"DEBUG: Raw prediction index: {prediction_index} (Type: {type(prediction_index)})"
        )
        # --- END MORE DEBUGGING ---

        prediction_name = "Fallback Default"  # Initialize with a different default

        # Map the index to the human-readable class name
        try:
            # Explicitly cast to int just in case it's a float like 0.0
            index = int(prediction_index)
            prediction_name = CLASS_NAMES[index]
            print(
                f"DEBUG: Mapped index {index} to name '{prediction_name}' successfully."
            )  # DEBUG Success

        except (IndexError, TypeError, ValueError) as e:  # Catch more potential errors
            print(
                f"Warning: Failed to map index {prediction_index}. Error: {type(e).__name__} - {e}"
            )
            prediction_name = "Unknown"  # Set to Unknown on failure

        # --- MORE DEBUGGING ---
        print(f"DEBUG: Returning prediction_name: '{prediction_name}'")
        # --- END MORE DEBUGGING ---

        return prediction_name

# src/predictor.py
import joblib
import pandas as pd
from pathlib import Path
from . import features  # Import feature extraction functions


class ActivityPredictor:
    """
    Handles loading a trained model and making predictions on new data windows.
    """

    def __init__(self, model_path: Path):
        """
        Loads the trained model from the specified path.

        :param model_path: Path object pointing to the .joblib model file.
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            print("Please run the 'train' command first to create the model.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # This list MUST match the feature names used during training (X_mean, Y_mean, etc.)
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

    def predict(self, data_window: list[dict]) -> int:
        """
        Predicts the activity for a single window of sensor data.

        :param data_window: A list of dictionaries, e.g., [{'X': 1, 'Y': 2, 'Z': 3}, ...]
        :return: The predicted label (0 for 'Stillness', 1 for 'Movement').
        """

        # 1. Convert window to a DataFrame (now has columns 'X', 'Y', 'Z')
        window_df = pd.DataFrame(data_window).astype(float)

        # 2. Extract features for each axis
        feature_dict = {}
        # Loop over uppercase axes ('X', 'Y', 'Z')
        for axis in ["X", "Y", "Z"]:
            # Create feature names like 'X_mean', 'Y_std', etc.
            feature_dict[f"{axis}_mean"] = features.calculate_mean(window_df[axis])
            feature_dict[f"{axis}_std"] = features.calculate_std_dev(window_df[axis])
            feature_dict[f"{axis}_rms"] = features.calculate_rms(window_df[axis])

        # 3. Create a DataFrame in the correct feature order
        features_df = pd.DataFrame([feature_dict], columns=self.feature_order)

        # 4. Make the prediction
        prediction = self.model.predict(features_df)

        return prediction[0]

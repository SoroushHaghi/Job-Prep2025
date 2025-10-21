# src/predictor.py
import joblib
import pandas as pd
from pathlib import Path
from . import features  # Import feature extraction functions from our package


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
            # Handle the case where the model doesn't exist yet
            print(f"Error: Model not found at {model_path}")
            print("Please run the 'train' command first to create the model.")
            raise
        except Exception as e:
            # Catch other potential loading errors (e.g., corrupt file)
            print(f"Error loading model: {e}")
            raise

        # This is the exact order of features the model was trained on.
        # It's crucial for prediction that the input DataFrame matches this order.
        self.feature_order = [
            "acc_x_mean",
            "acc_x_std",
            "acc_x_rms",
            "acc_y_mean",
            "acc_y_std",
            "acc_y_rms",
            "acc_z_mean",
            "acc_z_std",
            "acc_z_rms",
        ]

    def predict(self, data_window: list[dict]) -> int:
        """
        Predicts the activity for a single window of sensor data.

        :param data_window: A list of dictionaries, where each dict is a sensor sample
                            (e.g., {'x': 1.0, 'y': 0.5, 'z': 0.1})
        :return: The predicted label (e.g., 0 for 'Stillness', 1 for 'Movement').
        """

        # 1. Convert the list of dictionaries into a DataFrame
        # Ensure data is numeric (float) for calculations
        window_df = pd.DataFrame(data_window).astype(float)

        # 2. Extract features for each axis using the functions from features.py
        feature_dict = {}
        for axis in ["x", "y", "z"]:
            feature_dict[f"acc_{axis}_mean"] = features.calculate_mean(window_df[axis])
            feature_dict[f"acc_{axis}_std"] = features.calculate_std_dev(
                window_df[axis]
            )
            feature_dict[f"acc_{axis}_rms"] = features.calculate_rms(window_df[axis])

        # 3. Create a single-row DataFrame from the extracted features
        # It's wrapped in a list [feature_dict] to make it a single row.
        # We explicitly set the 'columns' to match self.feature_order.
        features_df = pd.DataFrame([feature_dict], columns=self.feature_order)

        # 4. Make the prediction
        # model.predict() returns an array (e.g., [1]), so we extract the first item.
        prediction = self.model.predict(features_df)

        return prediction[0]

# src/predictor.py
import joblib
import pandas as pd
from pathlib import Path
import torch  # --- NEW ---
from . import features

# --- NEW: Import your CNN model class definition ---
# TODO: You MUST import your defined CNN model class here.
# It's likely in a file like 'src/model.py'.
# from .model import CNNModel

# --- PLACEHOLDER: Remove this when you add the real import ---
# This is just so the file can be read without errors.
# Replace this with your actual CNNModel class.
try:
    from .model import CNNModel
except ImportError:
    print(
        "WARNING: src.model.CNNModel not found. "
        "Using a placeholder. CNN prediction WILL fail."
    )

    class CNNModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("--- THIS IS A PLACEHOLDER CNN CLASS ---")

        def forward(self, x):
            raise NotImplementedError(
                "Please import your actual CNNModel class from .model"
            )


# --- END PLACEHOLDER ---


# Define the class names (MUST match training order)
# RandomForest (throwing=0, drinking=1, driving=2)
# CNN (Stillness=0, Throwing=1, Drinking=2) -> Check your training!
# We will use the RF order for now, but be aware of this.
CLASS_NAMES = ["throwing", "drinking", "driving"]
# TODO: If your CNN model was trained with a different order
# (e.g., "stillness", "throwing", "drinking"), you will
# need to adjust the CLASS_NAMES list or the mapping logic.
# For now, we assume both models map 0, 1, 2 to the same classes.


class ActivityPredictor:
    """
    Handles loading a model (RF or CNN) and making predictions.
    """

    # --- MODIFIED: __init__ now takes model_type ---
    def __init__(self, model_path: Path, model_type: str):
        """
        Loads the trained model based on its type ('rf' or 'cnn').
        """
        self.model_type = model_type
        self.model = None

        try:
            if self.model_type == "rf":
                # --- RandomForest Loading Logic ---
                self.model = joblib.load(model_path)
                print(f"RandomForest model loaded from {model_path}")
                # Feature order is specific to RandomForest
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

            elif self.model_type == "cnn":
                # --- CNN Loading Logic ---
                # TODO: IMPORTANT!
                # You MUST instantiate your model class first, exactly
                # as you did in your training script (e.g., with
                # correct input_channels and num_classes).
                # Adjust this line to match your CNNModel's __init__
                self.model = CNNModel(input_channels=3, num_classes=3)

                # Load the saved weights (state dictionary)
                self.model.load_state_dict(torch.load(model_path))

                # Set model to evaluation mode (disables dropout, etc.)
                self.model.eval()
                print(f"CNN model loaded from {model_path}")

            else:
                raise ValueError(
                    f"Unknown model_type: '{model_type}'. " "Must be 'rf' or 'cnn'."
                )

        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            print("Please run the 'train' or 'train-dl' command first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # --- MODIFIED: predict() now has conditional logic ---
    def predict(self, data_window: list[dict]) -> str:
        """
        Predicts the activity for a window using the loaded model.
        Returns the class name.
        """
        # --- 1. Common Data Prep: Convert list[dict] to DataFrame ---
        window_df = pd.DataFrame(data_window).astype(float)

        prediction_index = -1  # Default to an invalid index

        # --- 2. Model-Specific Prediction Logic ---
        if self.model_type == "rf":
            # --- RandomForest Path (Feature Extraction) ---
            feature_dict = {}
            for axis in ["X", "Y", "Z"]:
                feature_dict[f"{axis}_mean"] = features.calculate_mean(window_df[axis])
                feature_dict[f"{axis}_std"] = features.calculate_std_dev(
                    window_df[axis]
                )
                feature_dict[f"{axis}_rms"] = features.calculate_rms(window_df[axis])

            features_df = pd.DataFrame([feature_dict], columns=self.feature_order)

            # Predict the class index
            prediction_index = self.model.predict(features_df)[0]

        elif self.model_type == "cnn":
            # --- CNN Path (Raw Data to Tensor) ---

            # 1. Get raw X, Y, Z data as a NumPy array
            # Shape will be (window_size, 3)
            raw_data = window_df[["X", "Y", "Z"]].values

            # 2. Transpose to get (num_channels, window_size)
            # Shape: (3, window_size)
            raw_data_transposed = raw_data.T

            # 3. Convert to PyTorch Tensor
            # Shape: (3, window_size)
            input_tensor = torch.tensor(raw_data_transposed, dtype=torch.float32)

            # 4. Add the batch dimension (PyTorch expects batch_size)
            # Shape: (1, 3, window_size)
            input_tensor = input_tensor.unsqueeze(0)

            # 5. Make prediction (in no_grad mode for efficiency)
            with torch.no_grad():
                # Output is raw logits, shape (1, num_classes)
                output_logits = self.model(input_tensor)

            # 6. Get the index of the highest logit
            # .item() converts the 1-element tensor to a Python int
            prediction_index = torch.argmax(output_logits, dim=1).item()

        # --- 3. Common Mapping: Map index to class name ---
        prediction_name = "Unknown"
        try:
            index = int(prediction_index)
            if 0 <= index < len(CLASS_NAMES):
                prediction_name = CLASS_NAMES[index]
            else:
                print(
                    f"Warning: Model ({self.model_type}) predicted an "
                    f"out-of-bounds index: {prediction_index}"
                )
        except (TypeError, ValueError) as e:
            print(
                f"Warning: Could not convert prediction index "
                f"'{prediction_index}' to int. Error: {e}"
            )

        return prediction_name

# src/dataset_builder.py
import numpy as np
import pandas as pd
import os

# --- ADJUST THIS IMPORT ---
# Import your feature functions from their module.
# I'm assuming your file is 'src/feature_extractor.py'.
# --------------------------
from feature_extractor import (
    calculate_mean,
    calculate_std_dev,
    calculate_rms,
    # Add all your other feature functions here (e.g., calculate_variance)
)


# --- Configuration ---
WINDOW_SIZE = 100  # The number of data points in each window
MOVEMENT_DATA_PATH = os.path.join("data", "movement_sample.csv")
OUTPUT_DATA_PATH = os.path.join("data", "features.csv")
# Generate enough stillness data to be comparable to movement data
STILLNESS_SAMPLES = 20000


def generate_stillness_data(num_samples, noise_level=0.05):
    """
    Generates data simulating a sensor at rest.
    For simplicity, we assume all three axes have a slight noise around zero.
    """
    x = np.random.normal(0, noise_level, num_samples)
    y = np.random.normal(0, noise_level, num_samples)
    z = np.random.normal(0, noise_level, num_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def process_data_windows(data, window_size):
    """
    Slides a window over the data and extracts features for each window.
    Returns a DataFrame where each row is a feature set for one window.
    """
    features_list = []

    # Iterate over the data in steps of WINDOW_SIZE
    for i in range(0, len(data) - window_size + 1, window_size):
        # This slice format is intentional for `black` compatibility
        window = data.iloc[i : i + window_size]

        # Ensure window is full size (drops the last partial window)
        if len(window) < window_size:
            continue

        window_features = {}
        for axis in ["X", "Y", "Z"]:
            # Apply each feature extraction function to each axis
            window_features[f"{axis}_mean"] = calculate_mean(window[axis])
            window_features[f"{axis}_std"] = calculate_std_dev(window[axis])
            window_features[f"{axis}_rms"] = calculate_rms(window[axis])

            # --- ADD YOUR OTHER FUNCTIONS HERE ---
            # Example:
            # window_features[f'{axis}_variance'] = calculate_variance(window[axis])
            # window_features[f'{axis}_min'] = calculate_min(window[axis])
            # window_features[f'{axis}_max'] = calculate_max(window[axis])

        features_list.append(window_features)

    return pd.DataFrame(features_list)


def build_feature_dataset():
    """
    Main function to build and save the complete feature dataset.
    """
    print("Starting dataset building process...")

    # 1. Process "Movement" data
    try:
        movement_data = pd.read_csv(MOVEMENT_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {MOVEMENT_DATA_PATH}")
        print("Please ensure 'movement_sample.csv' exists in the 'data' folder.")
        return
    except Exception as e:
        print(f"Error reading movement data: {e}")
        return

    movement_features = process_data_windows(movement_data, WINDOW_SIZE)
    movement_features["label"] = 1  # Label 1 for movement
    print(f"Processed {len(movement_features)} 'movement' windows.")

    # 2. Process "Stillness" data
    stillness_data = generate_stillness_data(STILLNESS_SAMPLES)
    stillness_features = process_data_windows(stillness_data, WINDOW_SIZE)
    stillness_features["label"] = 0  # Label 0 for stillness
    print(f"Processed {len(stillness_features)} 'stillness' windows.")

    # 3. Combine and save
    final_dataset = pd.concat(
        [movement_features, stillness_features], ignore_index=True
    )

    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)

    final_dataset.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nSuccessfully built and saved feature dataset to {OUTPUT_DATA_PATH}")
    print(f"Total samples (windows): {len(final_dataset)}")
    print("Dataset head:")
    print(final_dataset.head())


if __name__ == "__main__":
    # This allows you to run the script directly from the terminal
    # using: python src/dataset_builder.py
    build_feature_dataset()

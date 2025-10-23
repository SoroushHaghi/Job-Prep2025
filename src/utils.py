# src/utils.py
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd  # --- NEW ---
from pathlib import Path  # --- NEW ---
import warnings  # --- NEW ---


def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array.
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def apply_butterworth_filter(data, cutoff_freq, sampling_rate, order=4):
    """
    Applies a low-pass Butterworth filter to the data.
    """
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def detect_events(data, threshold):
    """
    Detects events in a signal where the absolute value
    exceeds a given threshold.

    Returns the indices of the detected events.
    """
    # Calculate the mean to center the data for easier detection
    mean_val = np.mean(data)
    centered_data = data - mean_val

    # Find all points where the absolute value is above the threshold
    event_indices = np.where(np.abs(centered_data) > threshold)[0]
    return event_indices


# ---- NEW CODE ADDED BELOW ----

# --- Constants for all modules ---
# We define them here so they can be imported by any other script.

# Defines the base directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

# Main window size for feature engineering and prediction
# MUST match the window size used in dataset_builder.py
WINDOW_SIZE = 10

# Defines the mapping from class name (folder/file name) to integer
# MUST match the order used in RandomForest training
CLASS_MAP = {
    "throwing": 0,
    "drinking": 1,
    "driving": 2,
}

# --- New Function for Deep Learning Data Loading ---


def load_and_segment_all(data_dir: Path, window_size: int):
    """
    Loads all raw CSV files from the CLASS_MAP, segments them,
    and returns data (as DataFrames) and labels.
    """
    all_data = []  # List to hold DataFrames
    all_labels = []  # List to hold corresponding integer labels

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for class_name, label_index in CLASS_MAP.items():
        file_path = data_dir / f"{class_name}.csv"

        if not file_path.exists():
            warnings.warn(
                f"Warning: Data file not found for class '{class_name}'"
                f" at {file_path}"
            )
            continue

        try:
            df = pd.read_csv(file_path)

            if not all(col in df.columns for col in ["X", "Y", "Z"]):
                warnings.warn(
                    f"Warning: File {file_path} is missing 'X', 'Y', 'Z' "
                    "columns. Skipping."
                )
                continue

            # In train_dl.py, we call this with window_size=1
            # This special case loads the *entire* file as one "segment"
            if window_size == 1:
                all_data.append(df)
                all_labels.append(label_index)
            else:
                # This is the standard segmentation logic
                for i in range(0, len(df) - window_size + 1, window_size):
                    window_df = df.iloc[i : i + window_size].copy()
                    all_data.append(window_df)
                    all_labels.append(label_index)

        except pd.errors.EmptyDataError:
            warnings.warn(f"Warning: File {file_path} is empty. Skipping.")
        except Exception as e:
            warnings.warn(f"Error processing file {file_path}: {e}. Skipping.")

    if not all_data:
        raise ValueError(
            f"No data was successfully loaded from {data_dir}. " "Check raw data files."
        )

    return all_data, all_labels

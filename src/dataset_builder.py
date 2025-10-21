# src/dataset_builder.py
import pandas as pd
from pathlib import Path
from . import features  # Import feature extraction functions
from .app import WINDOW_SIZE  # Use the same window size

# --- Configuration for Multi-Class Classification (3-Class) ---

# Define the numeric labels for our classes
# We start labels from 0
CLASS_LABELS = {
    "throwing": 0,
    "drinking": 1,
    "driving": 2,
}

# --- THIS SECTION IS UPDATED ---
# Map our activity files to their corresponding labels
# using the new, cleaner filenames.
ACTIVITY_FILES = {
    Path("data/throwing.csv"): CLASS_LABELS["throwing"],
    Path("data/drinking.csv"): CLASS_LABELS["drinking"],
    Path("data/driving.csv"): CLASS_LABELS["driving"],
}
# --- END OF UPDATE ---

# Output file remains the same
FEATURE_FILE = Path("data/features.csv")
# ---


def create_features_from_file(filepath: Path, label: int) -> pd.DataFrame:
    """
    Reads a raw data file, processes it into windows, extracts features,
    and assigns a label.
    """
    print(f"Processing file: {filepath} for label {label}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please make sure all activity CSVs are in the 'data/' directory.")
        return pd.DataFrame()  # Return empty DataFrame

    # --- Data Standardization (Robust column renaming) ---
    # This handles 'x', 'acc_x', and 'X' all mapping to 'X'
    df.rename(
        columns={
            "x": "X",
            "y": "Y",
            "z": "Z",
            "acc_x": "X",
            "acc_y": "Y",
            "acc_z": "Z",
        },
        inplace=True,
        errors="ignore",
    )

    # Ensure we only have the columns we need
    if not all(col in df.columns for col in ["X", "Y", "Z"]):
        print(f"Error: File {filepath} does not contain required X, Y, Z columns.")
        return pd.DataFrame()

    df = df[["X", "Y", "Z"]].dropna()
    # ---

    window_features = []

    # Process the file in windows
    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[i : i + WINDOW_SIZE]

        feature_dict = {}
        for axis in ["X", "Y", "Z"]:
            # Use the feature extraction functions
            feature_dict[f"{axis}_mean"] = features.calculate_mean(window[axis])
            feature_dict[f"{axis}_std"] = features.calculate_std_dev(window[axis])
            feature_dict[f"{axis}_rms"] = features.calculate_rms(window[axis])

        window_features.append(feature_dict)

    if not window_features:
        print(f"Warning: No feature windows generated for {filepath}.")
        return pd.DataFrame()

    features_df = pd.DataFrame(window_features)
    features_df["label"] = label  # Assign the correct label

    print(f"Processed {len(features_df)} windows for label {label}.")
    return features_df


def build_feature_dataset():
    """
    Main function to build the multi-class feature dataset.
    """
    print("Starting 3-class dataset building process...")

    all_features_list = []

    # Loop through and process all activity files
    for filepath, label in ACTIVITY_FILES.items():
        activity_features = create_features_from_file(filepath, label)
        if not activity_features.empty:
            all_features_list.append(activity_features)

    if not all_features_list:
        print("Error: No data processed. Feature file not created.")
        return

    # 3. Combine all features into one DataFrame
    final_dataset = pd.concat(all_features_list, ignore_index=True)

    # 4. Save the combined dataset
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(FEATURE_FILE, index=False)

    print(f"\nSuccessfully built and saved 3-class feature dataset to {FEATURE_FILE}")
    print(f"Total samples (windows): {len(final_dataset)}")
    print("Class distribution:")
    # Show value counts with human-readable labels
    label_map = {v: k for k, v in CLASS_LABELS.items()}
    print(final_dataset["label"].map(label_map).value_counts())

    print("\nDataset head:")
    print(final_dataset.head())
    print("Feature dataset built successfully.")

# src/visualize_data.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_raw_sensor_data(file_path: str, output_path: str):
    """
    Loads raw sensor data from a CSV file, plots the X, Y, and Z accelerometer data,
    and saves the plot to a file.
    """
    print(f"Loading raw sensor data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Generating raw sensor data plot...")
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df["X"], label="X")
        plt.plot(df["Y"], label="Y")
        plt.plot(df["Z"], label="Z")
        plt.title(f"Raw Sensor Data from {Path(file_path).name}")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        print(f"Raw data plot saved to {output_file}")
        plt.close()

    except Exception as e:
        print(f"Error generating or saving plot: {e}")

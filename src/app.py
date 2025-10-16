# src/app.py
import numpy as np
from .drivers import BaseSensorDriver

# Import your new feature calculation functions
from .features import calculate_mean, calculate_std_dev, calculate_rms


def run_processing_loop(driver: BaseSensorDriver, window_size: int):
    """
    Runs the main data processing loop, collecting data into windows
    and calculating features for each window.
    """
    print("--> Starting sensor data processing loop...")

    # Use a list to store the data for the current window
    window = []

    while True:
        sensor_data = driver.read()
        if sensor_data is None:
            print("--> End of data stream.")
            break

        # For this example, we'll focus on the 'x' axis data.
        window.append(sensor_data["x"])

        # If the window is full, process it
        if len(window) == window_size:
            # Convert the window to a NumPy array for efficient calculation
            window_np = np.array(window)

            # Calculate features for this window
            mean = calculate_mean(window_np)
            std_dev = calculate_std_dev(window_np)
            rms = calculate_rms(window_np)

            print("--- Window Features ---")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std Dev: {std_dev:.4f}")
            print(f"  RMS Energy: {rms:.4f}")
            print("-----------------------")

            # Clear the window to start collecting the next batch
            window = []

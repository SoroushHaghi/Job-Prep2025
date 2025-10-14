# sandbox/visualize_filter_effect.py
import matplotlib.pyplot as plt
import numpy as np

# Import the tools we built
from src.data_processing.reader import MockSensor
from src.utils import apply_butterworth_filter


def plot_filter_comparison():
    """
    Loads data from the mock sensor, applies a filter,
    and plots both signals for comparison.
    """
    print("Loading sensor data...")
    sensor = MockSensor()
    if not sensor.data:
        print("Could not load data. Aborting.")
        return

    # Extract the raw signal (Z-axis) and create a time axis
    raw_signal = np.array([reading[3] for reading in sensor.data])
    sampling_rate = 100  # We assume 100 Hz
    time_axis = np.arange(len(raw_signal)) / sampling_rate

    print("Applying Butterworth filter...")
    # Apply the filter to the raw signal
    filtered_signal = apply_butterworth_filter(
        data=raw_signal, cutoff_freq=5, sampling_rate=sampling_rate
    )

    print("Generating plot...")
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal, label="Raw Sensor Data", alpha=0.7)
    plt.plot(time_axis, filtered_signal, label="Filtered Data", linewidth=2)

    # Add labels and title for clarity
    plt.title("Effect of Butterworth Filter on Real Sensor Data")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    plot_filter_comparison()
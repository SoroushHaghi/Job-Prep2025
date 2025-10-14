# sandbox/visualize_event_detection.py
import matplotlib.pyplot as plt
import numpy as np

# Import our tools
from src.data_processing.reader import MockSensor
from src.utils import apply_butterworth_filter, detect_events


def plot_event_detection():
    """
    Loads, filters, and analyzes sensor data to detect and plot events.
    """
    # --- Step 1: Load and Filter Data (same as before) ---
    sensor = MockSensor()
    raw_signal = np.array([reading[3] for reading in sensor.data])
    sampling_rate = 100
    time_axis = np.arange(len(raw_signal)) / sampling_rate
    filtered_signal = apply_butterworth_filter(
        data=raw_signal, cutoff_freq=5, sampling_rate=sampling_rate
    )

    # --- Step 2: Analyze for Events ---
    # We'll set a threshold of 0.01 m/s^2. Any deviation larger than this
    # from the mean will be considered a "bump" or "shake".
    event_threshold = 0.01
    event_indices = detect_events(filtered_signal, threshold=event_threshold)
    print(f"Detected {len(event_indices)} event points.")

    # --- Step 3: Visualize the Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, filtered_signal, label="Filtered Data")

    # Mark the detected events on the plot with red vertical lines
    if len(event_indices) > 0:
        event_times = time_axis[event_indices]
        plt.vlines(
            event_times,
            ymin=plt.ylim()[0],
            ymax=plt.ylim()[1],
            color="red",
            linestyle="--",
            label=f"Events (Threshold > {event_threshold})",
        )

    plt.title("Event Detection in Filtered Sensor Data")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_event_detection()

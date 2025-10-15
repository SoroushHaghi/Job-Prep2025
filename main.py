# main.py
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.data_processing.reader import MockSensor
from src.utils import apply_butterworth_filter, detect_events


def run_filter_visualization():
    """Loads, filters, and plots the data."""
    print("--- Running Filter Visualization ---")
    sensor = MockSensor()
    raw_signal = np.array([reading[3] for reading in sensor.data])
    time_axis = np.arange(len(raw_signal)) / 100

    filtered_signal = apply_butterworth_filter(
        data=raw_signal, cutoff_freq=5, sampling_rate=100
    )

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_signal, label="Raw Sensor Data", alpha=0.7)
    plt.plot(time_axis, filtered_signal, label="Filtered Data", linewidth=2)
    plt.title("Effect of Butterworth Filter")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_event_detection(threshold):
    """Loads, filters, and detects events in the data."""
    print(f"--- Running Event Detection (Threshold: {threshold}) ---")
    sensor = MockSensor()
    raw_signal = np.array([reading[3] for reading in sensor.data])
    time_axis = np.arange(len(raw_signal)) / 100

    filtered_signal = apply_butterworth_filter(
        data=raw_signal, cutoff_freq=5, sampling_rate=100
    )
    event_indices = detect_events(filtered_signal, threshold=threshold)
    print(f"Found {len(event_indices)} event points.")

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, filtered_signal, label="Filtered Data")
    if len(event_indices) > 0:
        plt.vlines(
            time_axis[event_indices],
            ymin=plt.ylim()[0],
            ymax=plt.ylim()[1],
            color="red",
            linestyle="--",
            label="Events",  # <-- THE FIX IS HERE (removed the 'f')
        )
    plt.title("Event Detection in Sensor Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool for processing and analyzing sensor data."
    )
    parser.add_argument(
        "--plot-filter",
        action="store_true",
        help="Visualize the effect of the Butterworth filter.",
    )
    parser.add_argument(
        "--detect-events",
        type=float,
        metavar="THRESHOLD",
        help="Detect events in the signal with a given threshold.",
    )

    args = parser.parse_args()

    if args.plot_filter:
        run_filter_visualization()
    elif args.detect_events is not None:
        run_event_detection(threshold=args.detect_events)
    else:
        print("No action specified. Use --help to see available commands.")

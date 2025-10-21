# tests/test_utils.py
import numpy as np
from src.utils import moving_average, apply_butterworth_filter, detect_events
from src.signal_generator import generate_synthetic_signal

# Corrected import path based on previous fixes
from src.drivers import MockSensor


def test_moving_average_simple():
    # Arrange
    test_data = np.array([1, 2, 3, 4, 5])  # Use a numpy array for consistency
    window = 3
    expected = np.array([2.0, 3.0, 4.0])  # Expected result is also a numpy array

    # Act
    result = moving_average(test_data, window)

    # Assert
    # Use the correct numpy function to compare arrays element by element
    np.testing.assert_array_equal(result, expected)


def test_apply_butterworth_filter_reduces_noise():
    """
    Tests if the filter actually reduces the variance (noise) of a signal.
    """
    # Arrange
    time, noisy_signal = generate_synthetic_signal(
        duration_s=5,
        sampling_rate_hz=100,
        freq_hz=2,
        amplitude=1.0,
        noise_amplitude=0.5,
    )

    # Act
    filtered_signal = apply_butterworth_filter(
        data=noisy_signal, cutoff_freq=10, sampling_rate=100
    )

    # Assert
    assert np.var(filtered_signal) < np.var(noisy_signal)


def test_butterworth_filter_on_real_sensor_data():
    """
    Tests if the filter reduces noise on a real dataset from the MockSensor.
    """
    # Arrange

    # Correct argument name for MockSensor
    sensor = MockSensor(file_path="data/throwing.csv")

    assert len(sensor.data) > 0, "MockSensor should load data for the test."

    # --- THIS IS THE FIX ---
    # MockSensor now returns (X, Y, Z), so Z is at index 2
    raw_z_axis_data = [reading[2] for reading in sensor.data]
    # --- END OF FIX ---

    raw_signal = np.array(raw_z_axis_data)

    # Act
    filtered_signal = apply_butterworth_filter(
        data=raw_signal, cutoff_freq=5, sampling_rate=100
    )

    # Assert
    assert np.var(filtered_signal) < np.var(raw_signal)


def test_detect_events_no_events_found():
    """
    Tests that no events are detected in a stable signal.
    """
    # Arrange: A perfectly flat signal and a threshold of 0.1
    stable_signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    threshold = 0.1

    # Act: Run the detection
    event_indices = detect_events(stable_signal, threshold=threshold)

    # Assert: The result should be an empty list
    assert len(event_indices) == 0


def test_detect_events_finds_spikes():
    """
    Tests that events are correctly detected when a signal crosses the threshold.
    """
    # Arrange: A signal with two clear spikes at index 2 and 5
    signal_with_spikes = np.array([0, 0, 5, 0, 0, -5, 0])
    threshold = 4.0

    # Act: Run the detection
    event_indices = detect_events(signal_with_spikes, threshold=threshold)

    # Assert: The function should find the spikes at indices 2 and 5
    np.testing.assert_array_equal(event_indices, [2, 5])

# tests/test_utils.py
import numpy as np
from src.utils import moving_average, apply_butterworth_filter
from src.signal_generator import generate_synthetic_signal
from src.data_processing.reader import MockSensor


def test_moving_average_simple():
    # Arrange
    test_data = [1, 2, 3, 4, 5]
    window = 3
    expected = [2.0, 3.0, 4.0]

    # Act
    result = moving_average(test_data, window)

    # Assert
    assert result == expected


# IMPORTANT: Make sure there are TWO empty lines above this function definition.
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
        noise_amplitude=0.5
    )

    # Act
    filtered_signal = apply_butterworth_filter(
        data=noisy_signal,
        cutoff_freq=10,
        sampling_rate=100
    )

    # Assert
    assert np.var(filtered_signal) < np.var(noisy_signal)


# IMPORTANT: Make sure there are TWO empty lines above this function definition.
def test_butterworth_filter_on_real_sensor_data():
    """
    Tests if the filter reduces noise on a real dataset from the MockSensor.
    """
    # Arrange
    sensor = MockSensor()
    assert len(sensor.data) > 0, "MockSensor should load data for the test."
    raw_z_axis_data = [reading[3] for reading in sensor.data]
    raw_signal = np.array(raw_z_axis_data)

    # Act
    filtered_signal = apply_butterworth_filter(
        data=raw_signal,
        cutoff_freq=5,
        sampling_rate=100
    )

    # Assert
    assert np.var(filtered_signal) < np.var(raw_signal)

# IMPORTANT: Make sure there is one single, empty line after this line.
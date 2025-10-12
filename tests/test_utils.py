
# tests/test_utils.py
import numpy as np
from src.utils import moving_average, apply_butterworth_filter
from src.signal_generator import generate_synthetic_signal


def test_moving_average_simple():
    # Arrange
    test_data = [1, 2, 3, 4, 5]
    window = 3
    expected = [2.0, 3.0, 4.0]

    # Act
    result = moving_average(test_data, window)

    # Assert
    assert result == expected

def test_apply_butterworth_filter_reduces_noise():
    """
    Tests if the filter actually reduces the variance (noise) of a signal.
    """
    # Arrange: Create a signal with significant noise
    time, noisy_signal = generate_synthetic_signal(
        duration_s=5,
        sampling_rate_hz=100,
        freq_hz=2,
        amplitude=1.0,
        noise_amplitude=0.5
    )

    # Act: Apply the filter
    filtered_signal = apply_butterworth_filter(
        data=noisy_signal,
        cutoff_freq=10,
        sampling_rate=100
    )

    # Assert: The variance (a measure of noise) should decrease
    assert np.var(filtered_signal) < np.var(noisy_signal)

# tests/test_features.py

import numpy as np
from src.features import calculate_mean, calculate_std_dev, calculate_rms


def test_calculate_mean():
    # Arrange: Create a simple window of data
    test_window = np.array([1, 2, 3, 4, 5])

    # Act: Run the function
    result = calculate_mean(test_window)

    # Assert: Check if the result is correct
    expected = 3.0
    assert result == expected


def test_calculate_std_dev():
    # Arrange
    test_window = np.array([1, 2, 3, 4, 5])

    # Act
    result = calculate_std_dev(test_window)

    # Assert
    expected = np.sqrt(2.0)  # The actual standard deviation is sqrt(2)
    assert np.isclose(result, expected)


def test_calculate_rms():
    # Arrange
    test_window = np.array([1, 2, 3, 4])  # Use a different window for variety

    # Act
    result = calculate_rms(test_window)

    # Assert (1^2 + 2^2 + 3^2 + 4^2 = 1+4+9+16 = 30. 30/4 = 7.5. sqrt(7.5) = 2.7386...)
    expected = 2.7386127875
    assert np.isclose(result, expected)

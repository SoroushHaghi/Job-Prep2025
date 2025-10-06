# tests/test_utils.py
from src.utils import moving_average

def test_moving_average_simple():
    # Arrange
    test_data = [1, 2, 3, 4, 5]
    window = 3
    expected = [2.0, 3.0, 4.0]

    # Act
    result = moving_average(test_data, window)

    # Assert
    assert result == expected
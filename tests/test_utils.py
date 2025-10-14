# tests/test_utils.py
import numpy as np
from src.utils import moving_average, apply_butterworth_filter
from src.signal_generator import generate_synthetic_signal
from src.data_processing.reader import MockSensor # <-- This is the key new import

# Add this import at the top of your file with the others
from src.data_processing.reader import MockSensor

# ... (your existing two test functions remain here) ...


# Add this new function at the end of the file
def test_butterworth_filter_on_real_sensor_data():
    """
    Tests if the filter reduces noise on a real dataset from the MockSensor.
    """
    # --- Arrange ---
    # 1. Initialize our virtual sensor
    sensor = MockSensor()
    assert len(sensor.data) > 0, "MockSensor should load data for the test."

    # 2. Read all the data points and extract just the Z-axis acceleration
    # We use the Z-axis because it should be relatively stable around 9.8 m/s^2 (gravity)
    # but with sensor noise and small movements.
    raw_z_axis_data = [reading[3] for reading in sensor.data]
    raw_signal = np.array(raw_z_axis_data)

    # --- Act ---
    # 3. Apply the same Butterworth filter to this real-world signal
    # We'll use a cutoff frequency of 5Hz, a reasonable value for this type of data.
    filtered_signal = apply_butterworth_filter(
        data=raw_signal,
        cutoff_freq=5,
        sampling_rate=100  # Assuming our data was sampled at 100Hz
    )

    # --- Assert ---
    # 4. The assertion is the same: the filter should reduce the signal's variance (noise)
    assert np.var(filtered_signal) < np.var(raw_signal)
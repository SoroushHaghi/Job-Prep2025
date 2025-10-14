# tests/test_signal_generator.py
from src.signal_generator import generate_synthetic_signal


def test_generate_synthetic_signal_output_shape():
    """
    Tests if the generated signal and time arrays have the correct
    number of samples.
    """
    # Arrange
    duration = 5  # 5 seconds
    sampling_rate = 100  # 100 Hz
    expected_samples = duration * sampling_rate

    # Act
    time, signal = generate_synthetic_signal(
        duration_s=duration,
        sampling_rate_hz=sampling_rate,
        freq_hz=2,
        amplitude=1.0,
        noise_amplitude=0.2,
    )

    assert len(time) == expected_samples
    assert len(signal) == expected_samples

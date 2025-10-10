
# src/signal_generator.py
import numpy as np


def generate_synthetic_signal(
    duration_s, sampling_rate_hz, freq_hz, amplitude, noise_amplitude
):
    """
    Generates a synthetic sine wave signal with added Gaussian noise.

    Args:
        duration_s (float): The duration of the signal in seconds.
        sampling_rate_hz (int): The number of samples per second.
        freq_hz (float): The frequency of the sine wave in Hertz.
        amplitude (float): The amplitude of the clean sine wave.
        noise_amplitude (float): The amplitude of the Gaussian noise.

    Returns:
        tuple: A tuple containing the time array and the noisy signal array.
    """
    num_samples = int(duration_s * sampling_rate_hz)
    time = np.linspace(0, duration_s, num_samples, endpoint=False)

    # Clean signal
    clean_signal = amplitude * np.sin(2 * np.pi * freq_hz * time)

    # Noise
    noise = noise_amplitude * np.random.normal(size=time.shape)

    # Final noisy signal
    noisy_signal = clean_signal + noise

    return time, noisy_signal

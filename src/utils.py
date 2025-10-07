# src/utils.py
from scipy.signal import butter, filtfilt
import numpy as np

def moving_average(data, window_size):
    """Calculates the moving average of a list of numbers."""
    if not data or window_size <= 0:
        return []

    averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        averages.append(sum(window) / window_size)
    return averages
def apply_butterworth_filter(data, cutoff_freq, sampling_rate, order=4):
    """
    Applies a low-pass Butterworth filter to the data.
    """
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
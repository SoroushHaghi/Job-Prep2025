# src/utils.py
import numpy as np
from scipy.signal import butter, filtfilt


def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array.
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def apply_butterworth_filter(data, cutoff_freq, sampling_rate, order=4):
    """
    Applies a low-pass Butterworth filter to the data.
    """
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def detect_events(data, threshold):
    """
    Detects events in a signal where the absolute value
    exceeds a given threshold.

    Returns the indices of the detected events.
    """
    # Calculate the mean to center the data around zero for easier detection
    mean_val = np.mean(data)
    centered_data = data - mean_val

    # Find all points where the absolute value is above the threshold
    event_indices = np.where(np.abs(centered_data) > threshold)[0]
    return event_indices
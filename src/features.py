# src/features.py

import numpy as np


def calculate_mean(window: np.ndarray) -> float:
    """Calculates the average value of a window."""
    return np.mean(window)


def calculate_std_dev(window: np.ndarray) -> float:
    """Calculates the standard deviation of a window."""
    return np.std(window)


def calculate_rms(window: np.ndarray) -> float:
    """
    Calculates the Root Mean Square of the signal window.
    This is a measure of the signal's energy or magnitude.
    """
    return np.sqrt(np.mean(np.square(window)))

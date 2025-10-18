# src/dataset_builder.py
import numpy as np
import pandas as pd


def generate_stillness_data(num_samples, noise_level=0.05):
    """
    Generates data simulating a sensor at rest.
    For simplicity, we assume all three axes have a slight noise around zero.
    """
    x = np.random.normal(0, noise_level, num_samples)
    y = np.random.normal(0, noise_level, num_samples)
    z = np.random.normal(0, noise_level, num_samples)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})

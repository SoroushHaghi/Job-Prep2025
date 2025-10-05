import pandas as pd
import numpy as np

# 1. Starting in the familiar world of Pandas
data = {
    'timestamp': [0.1, 0.2, 0.3, 0.4, 0.5],
    'signal_value': [5, 5.2, 5.1, 7.5, 5.3] # A noise spike
}
df = pd.DataFrame(data)
print("--- Pandas World ---")
print(df)
print("-" * 20)

# 2. Building the bridge to the NumPy world
signal_numpy_array = df['signal_value'].to_numpy()
print("\n--- NumPy World ---")
print(signal_numpy_array)
print(type(signal_numpy_array))
print("-" * 20)
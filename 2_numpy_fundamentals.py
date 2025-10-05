import numpy as np

# Task 2.1: Create NumPy arrays

# 1. From a Python list
list_array = np.array([1, 2, 3, 4, 5])
print(f"Array from list: {list_array}")
print(f"Type of this array: {type(list_array)}")
# 2. Array of zeros
zeros_array = np.zeros(5)
print(f"Array of zeros: {zeros_array}")

# 3. Array of ones
ones_array = np.ones(5)
print(f"Array of ones: {ones_array}")

# 4. A sequence of numbers with a step (arange)
arange_array = np.arange(0, 10, 2)
print(f"Array from arange: {arange_array}")

# 5. A specific number of evenly spaced points (linspace)
linspace_array = np.linspace(0, 1, 5)
print(f"Array from linspace: {linspace_array}")
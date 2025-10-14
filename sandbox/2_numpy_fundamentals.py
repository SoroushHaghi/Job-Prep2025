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

# --- Task 2.2: Vectorized Operations ---
print("\n--- 2.2: Vectorized Operations ---")
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# These operations happen element-by-element, no 'for' loop needed!
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")

# --- Task 2.3: Universal Functions (ufuncs) ---
print("\n--- 2.3: Universal Functions ---")
angles = np.array([0, np.pi / 2, np.pi])  # np.pi is a built-in constant for pi
print(f"Sine of angles: {np.sin(angles)}")

data = np.array([10, 20, 30, 40, 50])
print(f"Mean of data: {np.mean(data)}")

# --- Task 2.4: Indexing and Slicing ---
print("\n--- 2.4: Indexing & Slicing ---")
my_array = np.arange(10)  # Creates an array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original array: {my_array}")

# Get the third element (indexing starts at 0)
print(f"Element at index 2: {my_array[2]}")

# Get a 'slice' of the array from index 2 up to (but not including) 5
print(f"Slice from index 2 to 5: {my_array[2:5]}")

# A powerful trick: get all elements that meet a condition
print(f"Elements > 5: {my_array[my_array > 5]}")

import numpy as np
import matplotlib.pyplot as plt

# Task 3.1: Create a simulated signal

# Create a time axis: 1000 points from 0 to 10 seconds
time = np.linspace(0, 10, 1000)

# Create a clean 1 Hz sine wave signal on that time axis
frequency = 1
clean_signal = np.sin(2 * np.pi * frequency * time)

# Use Matplotlib to plot the signal
plt.plot(time, clean_signal)
plt.title("Our Perfect, Clean Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Task 3.2: Add noise to the signal
noise_amplitude = 0.5
noise = noise_amplitude * np.random.normal(size=time.shape)
noisy_signal = clean_signal + noise

# Plot both the clean and noisy signals
plt.figure(figsize=(12, 6)) # Make the plot a bit wider
plt.plot(time, noisy_signal, label='Noisy Signal')
plt.plot(time, clean_signal, label='Clean Signal', linewidth=3)
plt.title("Clean vs. Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

from scipy.signal import savgol_filter

# Task 3.3: Filter the noise
# Apply a Savitzky-Golay filter to smooth the noisy signal
filtered_signal = savgol_filter(noisy_signal, window_length=51, polyorder=3)

# Plot all three signals to compare
plt.figure(figsize=(12, 6))
plt.plot(time, noisy_signal, label='Noisy Signal', alpha=0.5)
plt.plot(time, filtered_signal, label='Filtered Signal', linewidth=3, color='green')
plt.plot(time, clean_signal, label='Original Clean Signal', linestyle='--', linewidth=2, color='black')
plt.title("Filtering a Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
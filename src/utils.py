# src/utils.py

def moving_average(data, window_size):
    """Calculates the moving average of a list of numbers."""
    if not data or window_size <= 0:
        return []

    averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        averages.append(sum(window) / window_size)
    return averages
# src/app.py

# Use an explicit relative import with a dot (.)
from .drivers import BaseSensorDriver


def run_processing_loop(driver: BaseSensorDriver):
    """
    Runs the main data processing loop.
    """
    print("--> Starting sensor data processing loop...")
    sample_count = 0
    while True:
        sensor_data = driver.read()
        if sensor_data is None:
            print("--> End of data stream.")
            break
        print(f"Read sample {sample_count}: {sensor_data}")
        sample_count += 1
    print(f"--> Processing finished. Total samples read: {sample_count}")

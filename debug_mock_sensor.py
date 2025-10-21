# debug_mock_sensor.py
print("--- Starting debug script ---")
try:
    # Try to import the MockSensor class
    from src.drivers import MockSensor

    print("Import successful!")

    # Try to create an instance, which should trigger the debug prints
    print("Instantiating MockSensor...")
    sensor = MockSensor(file_path="data/throwing.csv")
    print("Instantiation finished.")

    # Check if data was loaded
    print(f"Length of sensor.data: {len(sensor.data)}")
    if len(sensor.data) > 0:
        print("Data loaded successfully!")
    else:
        print("!!! Data failed to load. Check MockSensor logs. !!!")

except ImportError as e:
    print(f"!!! FAILED TO IMPORT MockSensor: {e} !!!")
except Exception as e:
    print(f"!!! An unexpected error occurred: {type(e).__name__} - {e} !!!")

print("--- Debug script finished ---")

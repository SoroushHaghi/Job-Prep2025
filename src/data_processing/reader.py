import csv
import time


class MockSensor:
    """
    A mock sensor class that simulates reading data from a real sensor
    by reading from a CSV file.
    """
    def __init__(self, file_path='data/accelerometer_data.csv'):
        """
        Initializes the sensor by loading the data from the CSV file.
        """
        self.file_path = file_path
        self.data = []
        self._load_data()
        self.current_index = 0

    def _load_data(self):
        """
        Private method to load all sensor readings from file into memory.
        """
        try:
            with open(self.file_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header without assigning to a variable
                for row in csv_reader:
                    # Convert string values to float for processing
                    timestamp = float(row[0])
                    acc_x = float(row[1])
                    acc_y = float(row[2])
                    acc_z = float(row[3])
                    self.data.append((timestamp, acc_x, acc_y, acc_z))
            print(f"Loaded {len(self.data)} points from {self.file_path}")
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {self.file_path}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def read_data(self):
        """
        Reads the next available data point from the loaded data.
        Returns None if all data has been read.
        """
        if self.current_index < len(self.data):
            data_point = self.data[self.current_index]
            self.current_index += 1
            return data_point
        else:
            print("End of dataset reached.")
            return None


# --- Example of how to use this class ---
if __name__ == "__main__":
    print("Initializing the mock sensor...")
    sensor = MockSensor()

    if sensor.data:  # Only proceed if data was loaded successfully
        print("\nReading first 5 data points:")
        for _ in range(5):
            reading = sensor.read_data()
            if reading:
                ts, x, y, z = reading
                # This print statement is now broken into multiple lines
                print(
                    f"Timestamp: {ts}, Accel(x,y,z): "
                    f"({x:.2f}, {y:.2f}, {z:.2f})"
                )

        print("\nSimulating real-time reading...")
        sensor.current_index = 0  # Reset index to simulate starting over
        while True:
            reading = sensor.read_data()
            if reading is None:
                break
            ts, x, y, z = reading
            print(f"Read: Accel X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
            time.sleep(0.1)  # Wait 100ms to simulate a 10Hz sensor
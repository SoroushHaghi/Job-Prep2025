# src/drivers.py
from abc import ABC, abstractmethod
import pandas as pd
import time  # Import time for the delay


# --- Base Class (no changes) ---
class BaseSensorDriver(ABC):
    """
    An abstract base class for sensor drivers. It defines the contract
    that all sensor drivers must follow.
    """

    @abstractmethod
    def read(self):
        """Reads a single data sample from the source."""
        pass


# --- SimulationDriver (This is what 'predict-stream' uses) ---
class SimulationDriver(BaseSensorDriver):
    """
    A data-driven simulation driver that reads sensor data from a CSV file
    and replays it row by row, ensuring consistent column names.
    """

    def __init__(self, filepath: str):
        dataframe = pd.read_csv(filepath).dropna()
        # Rename ALL possible column variations to the standard 'X', 'Y', 'Z'
        dataframe.rename(
            columns={
                "x": "X",
                "y": "Y",
                "z": "Z",
                "acc_x": "X",
                "acc_y": "Y",
                "acc_z": "Z",
            },
            inplace=True,
            errors="ignore",
        )
        self.iterator = dataframe.itertuples()

    def read(self) -> dict | None:
        """Reads the next row, returns dict with UPPERCASE keys."""
        try:
            row = next(self.iterator)
            return {"X": row.X, "Y": row.Y, "Z": row.Z}
        except StopIteration:
            return None
        except AttributeError as e:
            print(f"Error reading row: {e}")
            print("Ensure CSV has 'X,Y,Z' or 'x,y,z' or 'acc_x,acc_y,acc_z' columns.")
            return None


# --- MockSensor (Final Version - No Timestamp Needed for Test) ---
class MockSensor(BaseSensorDriver):
    """
    Mock sensor for testing. Reads X, Y, Z from CSV, ignores timestamp.
    """

    def __init__(self, file_path="data/throwing.csv", delay_s=0.01):
        self.delay_s = delay_s
        self.data = []
        try:
            df = pd.read_csv(file_path)
            df.dropna(inplace=True)

            # --- Robust Renaming (No change needed here) ---
            rename_map = {
                "x": "X",
                "y": "Y",
                "z": "Z",
                "acc_x": "X",
                "acc_y": "Y",
                "acc_z": "Z",
            }
            df.rename(columns=rename_map, inplace=True, errors="ignore")
            # --- End Renaming ---

            # --- MODIFIED: Check only for X, Y, Z ---
            required_cols = ["X", "Y", "Z"]
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise KeyError(
                    f"Missing required columns after standardization: Need {required_cols}, Missing {missing}, Found {list(df.columns)}"
                )

            # --- MODIFIED: Convert only X, Y, Z to list of tuples ---
            self.data = list(zip(df["X"], df["Y"], df["Z"]))  # No timestamp

        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR [MockSensor]: Failed to load {file_path}: {e}")
            self.data = []  # Ensure data is empty on failure
        except Exception as e:
            print(
                f"ERROR [MockSensor]: Unexpected error loading {file_path}: {type(e).__name__} - {e}"
            )
            self.data = []

        self.iterator = iter(self.data)

    def read(self):
        """Reads next sample (X, Y, Z), simulates delay, returns tuple or None."""
        try:
            time.sleep(self.delay_s)
            # --- MODIFIED: Returns tuple (X, Y, Z) ---
            return next(self.iterator)
        except StopIteration:
            return None

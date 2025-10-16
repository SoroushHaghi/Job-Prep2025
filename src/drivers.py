# src/drivers.py

from abc import ABC, abstractmethod
import pandas as pd


class BaseSensorDriver(ABC):
    """
    An abstract base class for sensor drivers. It defines the contract
    that all sensor drivers must follow.
    """

    @abstractmethod
    def read(self):
        """Reads a single data sample from the source."""
        pass


class SimulationDriver(BaseSensorDriver):
    """
    A data-driven simulation driver that reads sensor data from a CSV file
    and replays it row by row.
    """

    def __init__(self, filepath: str):
        """
        Initializes the driver by loading the dataset from the given file path.

        Args:
            filepath: The path to the CSV file containing the sensor data.
        """
        # Read the CSV and immediately drop any rows with missing values
        self.dataframe = pd.read_csv(filepath).dropna()
        self.iterator = self.dataframe.itertuples()

    def read(self) -> dict | None:
        """
        Reads the next row from the CSV file and returns it as a dictionary.

        Returns:
            A dictionary with sensor data (e.g., {'x': 1.0, 'y': 0.5, 'z': 9.8})
            or None if the end of the file is reached.
        """
        try:
            row = next(self.iterator)
            # Assumes the CSV has columns named 'X', 'Y', 'Z'.
            # The keys 'x', 'y', 'z' are lowercase for consistency in the application.
            return {"x": row.X, "y": row.Y, "z": row.Z}
        except StopIteration:
            # This happens when all data has been read
            return None

# src/main.py
import typer

from .drivers import SimulationDriver
from .app import run_processing_loop


def main(
    data_file: str = typer.Option(
        ...,
        "--data-file",
        "-f",
        help="Path to the sensor data CSV file.",
    ),
    window_size: int = typer.Option(
        50,
        "--window-size",
        "-w",
        # This formatting is clean and avoids all line-length issues.
        help="The number of samples to collect before calculating features.",
    ),
):
    """
    Processes sensor data from a given file by calculating features over sliding windows.
    """
    print(f"Initializing simulation with data from: {data_file}")
    print(f"Using window size: {window_size}")

    driver = SimulationDriver(filepath=data_file)
    run_processing_loop(driver=driver, window_size=window_size)

    print("Operation completed successfully.")


if __name__ == "__main__":
    typer.run(main)

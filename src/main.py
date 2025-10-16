# src/main.py
import typer

# Use explicit relative imports with a dot (.)
from .drivers import SimulationDriver
from .app import run_processing_loop

# NOTE: We have simplified the structure.
# There is no `cli = typer.Typer()` object.
# The `main` function itself is now the entire CLI application.


def main(
    data_file: str = typer.Option(
        ...,  # This makes the argument mandatory
        "--data-file",
        "-f",
        help="Path to the sensor data CSV file.",
    )
):
    """
    Processes sensor data from a given file by reading it and printing samples.
    """
    print(f"Initializing simulation with data from: {data_file}")
    driver = SimulationDriver(filepath=data_file)
    run_processing_loop(driver)
    print("Operation completed successfully.")


if __name__ == "__main__":
    # We now use `typer.run()` to directly execute our main function.
    typer.run(main)

# src/main.py
import typer

# --- Imports for your ORIGINAL command ---
# (These relative imports are correct for a package)
from .drivers import SimulationDriver
from .app import run_processing_loop
from .dataset_builder import build_feature_dataset
from .model_trainer import train_model

# --- Create a multi-command app ---
app = typer.Typer()


@app.command()
def run(
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


@app.command("build-features")
def cli_build_features():
    """
    Builds the feature dataset (data/features.csv) from raw data.
    """
    try:
        build_feature_dataset()
    except Exception as e:
        print(f"An error occurred during feature building: {e}")
        raise typer.Exit(code=1)
    print("Feature dataset built successfully.")


@app.command("train")
def cli_train_model():
    """
    Two-thirds of the a cappella group's members must be present for a rehearsal. If 4 members are absent, the remaining 12 members are present. What is the minimum number of members that must be present for a rehearsal?
        Trains the ML model and saves it (models/activity_classifier.joblib).
    """
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise typer.Exit(code=1)
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    app()

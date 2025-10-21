# src/main.py
import typer
from pathlib import Path  # Import Path for handling file paths
from rich import print  # For better, colored terminal output

# --- Imports for your ORIGINAL commands ---
from .drivers import SimulationDriver
from .app import run_processing_loop
from .dataset_builder import build_feature_dataset
from .model_trainer import train_model

# --- Imports for the NEW command ---
from .predictor import ActivityPredictor  # Import our new predictor class

# --- Create a multi-command app ---
app = typer.Typer(help="Main CLI for the Activity Recognition Project")


# --- Your Existing Commands (with fixes) ---


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
    Trains the ML model and saves it (models/activity_classifier.joblib).
    """
    # NOTE: The incorrect docstring about 'a cappella group' has been removed.
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise typer.Exit(code=1)
    print("Model trained and saved successfully.")


# --- NEW Command (from our previous step) ---


@app.command(help="Simulate a real-time stream and predict activity.")
def predict_stream(
    data_file: Path = typer.Option(
        ...,  # '...' makes this option mandatory
        "--data-file",
        "-f",
        help="Path to the simulation data file (e.g., data/processed/movement_sample.csv)",
        exists=True,  # Typer will check if the file exists
        readable=True,
        show_default=False,
    ),
    model_file: Path = typer.Option(
        "models/activity_classifier.joblib",  # Default model path
        "--model-file",
        "-m",
        help="Path to the trained model file (.joblib)",
        exists=True,
        readable=True,
    ),
    window_size: int = typer.Option(
        10,  # Default window size
        "--window-size",
        "-w",
        help="Number of samples to collect before making a prediction. MUST match training window size.",
    ),
):
    """
    Simulates a sensor stream, collects data in windows,
    and predicts the activity (Movement or Stillness) using the trained model.
    """
    print("📊 [bold]Starting Real-Time Activity Prediction[/bold]")
    print(f"🧠 Loading model from: [cyan]{model_file}[/cyan]")

    try:
        # 1. Initialize the "brain"
        predictor = ActivityPredictor(model_path=model_file)
    except FileNotFoundError:
        print("[bold red]Error: Model file not found.[/bold red]")
        raise typer.Exit(code=1)

    print(f"📡 Initializing sensor stream from: [cyan]{data_file}[/cyan]")
    print(f"⏱️ Using window size: [bold]{window_size}[/bold] samples")
    print("-" * 40)

    # 2. Initialize the "senses"
    # Make sure your SimulationDriver accepts 'filepath'
    driver = SimulationDriver(filepath=data_file)
    window = []  # This list will hold the current window of data

    # Map prediction codes (0, 1) to human-readable, colored labels
    label_map = {
        0: "[bold red]Stillness[/bold red]",
        1: "[bold green]Movement[/bold green]",
    }

    try:
        while True:
            # 3. Read one sample from the sensor simulator
            sample = driver.read()

            # 4. Check if the stream has ended
            if sample is None:
                # This line is corrected (no 'f')
                print("\n🏁 [yellow]End of sensor stream.[/yellow]")
                break

            # 5. Add the new sample to our window
            window.append(sample)

            # 6. Check if the window is full
            if len(window) == window_size:

                # 7. If full, make a prediction
                prediction_code = predictor.predict(window)
                prediction_label = label_map.get(
                    prediction_code, "[bold yellow]Unknown[/bold yellow]"
                )

                # 8. Print the result
                # THIS IS THE LINE THAT WAS CUT OFF AND CAUSED THE SYNTAX ERROR
                print(f"Prediction: {prediction_label}")

                # 9. Clear the window to start collecting the next batch
                window = []

    except KeyboardInterrupt:
        # Allow the user to stop the stream gracefully
        # This line is corrected (no 'f')
        print("\n🛑 [yellow]Stream stopped by user.[/yellow]")
    except Exception as e:
        # Catch any other unexpected errors during the loop
        # This line NEEDS the 'f' because it uses the {e} variable
        print(f"\n[bold red]An error occurred during the stream: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

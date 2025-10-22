# src/main.py
import typer
from pathlib import Path  # Import Path for handling file paths
from rich import print  # For better, colored terminal output

# --- Imports for project modules ---
from .drivers import SimulationDriver
from .app import run_processing_loop  # Removed unused WINDOW_SIZE import
from .dataset_builder import build_feature_dataset
from .model_trainer import train_model
from .predictor import ActivityPredictor  # Import our multi-class predictor

# --- Create a multi-command app ---
app = typer.Typer(help="Main CLI for the Activity Recognition Project")


# --- Existing Commands ---


@app.command()
def run(
    data_file: str = typer.Option(
        ..., "--data-file", "-f", help="Path to the sensor data CSV file."
    ),
    window_size: int = typer.Option(
        10,  # Standardized window size
        "--window-size",
        "-w",
        help="Number of samples before feature calculation.",
    ),
):
    """
    (Legacy Command) Processes sensor data, calculates features over windows.
    """
    print(f"Initializing simulation with data from: {data_file}")
    print(f"Using window size: {window_size}")
    try:
        driver = SimulationDriver(filepath=data_file)
        # Note: run_processing_loop might need update if it doesn't handle 'X','Y','Z'
        run_processing_loop(driver=driver, window_size=window_size)
        print("Operation completed successfully.")
    except Exception as e:
        print(f"[bold red]Error during run: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("build-features")
def cli_build_features():
    """
    Builds the multi-class feature dataset (data/features.csv).
    """
    try:
        build_feature_dataset()
    except Exception as e:
        print(f"[bold red]Error during feature building: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("train")
def cli_train_model():
    """
    Trains the multi-class ML model and saves it with confusion matrix.
    """
    try:
        train_model()
    except Exception as e:
        print(f"[bold red]Error during model training: {e}[/bold red]")
        raise typer.Exit(code=1)


# --- UPDATED Command with Colors ---


@app.command(help="Simulate real-time stream & predict activity (3-class).")
def predict_stream(
    data_file: Path = typer.Option(
        ...,
        "--data-file",
        "-f",
        help="Path to simulation CSV (e.g., data/throwing.csv)",
        exists=True,
        readable=True,
        show_default=False,
    ),
    model_file: Path = typer.Option(
        "models/activity_classifier.joblib",  # Correctly closed string
        "--model-file",
        "-m",
        help="Path to the trained multi-class model (.joblib)",
        exists=True,
        readable=True,
    ),
    window_size: int = typer.Option(
        10,  # Default window size matches training
        "--window-size",
        "-w",
        help="Samples per prediction window. MUST match training.",
    ),
):
    """
    Simulates sensor stream, predicts activity (throwing, drinking, driving).
    """
    # --- DEBUG LINE ADDED EARLIER ---
    # print("--- DEBUG: predict_stream function started ---")
    # --- END DEBUG ---
    print("📊 [bold]Starting Real-Time Activity Prediction[/bold]")
    print(f"🧠 Loading model from: [cyan]{model_file}[/cyan]")
    try:
        predictor = ActivityPredictor(model_path=model_file)
    except FileNotFoundError:
        print(
            f"[bold red]Error: Model file not found at {model_file}. Run 'train' first.[/bold red]"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[bold red]Error loading predictor: {e}[/bold red]")
        raise typer.Exit(code=1)

    print(f"📡 Initializing sensor stream from: [cyan]{data_file}[/cyan]")
    print(f"⏱️ Using window size: [bold]{window_size}[/bold] samples")
    print("-" * 40)

    try:
        driver = SimulationDriver(filepath=data_file)
    except Exception as e:
        print(f"[bold red]Error initializing simulation driver: {e}[/bold red]")
        raise typer.Exit(code=1)

    window = []

    # --- Logic with Rich Formatting Re-added ---
    try:
        while True:
            sample = driver.read()  # Reads {'X': ..., 'Y': ..., 'Z': ...}
            if sample is None:
                print("\n🏁 [yellow]End of sensor stream.[/yellow]")
                break

            window.append(sample)

            if len(window) == window_size:
                try:
                    # predictor.predict now returns the class name string directly
                    prediction_label = predictor.predict(window)

                    # --- Rich Formatting for 3 Classes ---
                    if prediction_label == "throwing":
                        formatted_label = f"[bold yellow]{prediction_label}[/bold yellow]"  # Yellow for throwing
                    elif prediction_label == "drinking":
                        formatted_label = f"[bold cyan]{prediction_label}[/bold cyan]"  # Cyan for drinking
                    elif prediction_label == "driving":
                        formatted_label = f"[bold magenta]{prediction_label}[/bold magenta]"  # Magenta for driving
                    else:  # Fallback for "Unknown"
                        formatted_label = f"[bold red]{prediction_label}[/bold red]"

                    print(f"Prediction: {formatted_label}")
                    # --- End Rich Formatting ---

                except Exception as e:
                    print(
                        f"[bold red]Error during prediction calculation: {e}[/bold red]"
                    )

                window = []  # Clear window for next batch

    # --- End Logic ---

    except KeyboardInterrupt:
        # Corrected print statement (no f-string needed) - Ensure this block is complete
        print(
            "\n🛑 [yellow]Stream stopped by user.[/yellow]"
        )  # <-- Make sure this line exists and is complete
    except Exception as e:
        # Corrected print statement (needs f-string for variable e)
        print(f"\n[bold red]An error occurred during the stream: {e}[/bold red]")
        raise typer.Exit(code=1)


# --- Ensure this block is present at the end ---
if __name__ == "__main__":
    app()
# --- End of file ---

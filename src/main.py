import typer
from pathlib import Path
from rich import print
import random  # <-- NEW: For random demo file selection
from typing import Optional  # <-- NEW: For optional parameters

# --- Imports for project modules ---
from .drivers import SimulationDriver
from .app import run_processing_loop
from .dataset_builder import build_feature_dataset
from .model_trainer import train_model
from .predictor import ActivityPredictor
from .visualize_data import plot_raw_sensor_data
from .plot_comparison import plot_comparison
from .train_dl import train_deep_learning_model

# --- Create a multi-command app ---
app = typer.Typer(help="Main CLI for the Activity Recognition Project")


@app.command("plot-comparison")
def cli_plot_comparison(
    rf_results_path: str = typer.Option(
        "models/rf_results.json",
        "--rf-results",
        help="Path to the RandomForest results JSON file.",
    ),
    cnn_results_path: str = typer.Option(
        "models/cnn_results.json",
        "--cnn-results",
        help="Path to the 1D-CNN results JSON file.",
    ),
    output_path: str = typer.Option(
        "docs/comparison_plot.png",
        "--output",
        help="Path to save the comparison plot.",
    ),
):
    """
    Generates a comparative plot of the F1-scores for the two models.
    """
    try:
        plot_comparison(rf_results_path, cnn_results_path, output_path)
    except Exception as e:
        print(f"[bold red]Error during plot comparison: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("visualize-data")
def cli_visualize_data(
    data_file: str = typer.Option(
        ..., "--data-file", "-f", help="Path to the raw sensor data CSV file."
    ),
    output_file: str = typer.Option(
        "docs/raw_data_visualization.png",
        "--output-file",
        "-o",
        help="Path to save the output plot.",
    ),
):
    """
    Generates a plot of the raw sensor data (X, Y, Z) from a CSV file.
    """
    try:
        plot_raw_sensor_data(file_path=data_file, output_path=output_file)
    except Exception as e:
        print(f"[bold red]Error during data visualization: {e}[/bold red]")
        raise typer.Exit(code=1)


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


@app.command("train-dl")
def cli_train_dl_model():
    """
    Trains the 1D-CNN Deep Learning model (cnn_model.pth).
    """
    try:
        train_deep_learning_model()
    except Exception as e:
        print(f"[bold red]Error during DL model training: {e}[/bold red]")
        raise typer.Exit(code=1)


# --- UPDATED predict_stream Command ---


@app.command(help="Simulate real-time stream & predict activity (3-class).")
def predict_stream(
    # --- MODIFIED: data_file is now Optional for demo mode ---
    data_file: Optional[Path] = typer.Option(
        None,
        "--data-file",
        "-f",
        help="Path to simulation CSV. If omitted, runs demo with a random file.",
        show_default=False,
    ),
    # --- MODIFIED: model_file is now Optional for auto-selection ---
    model_file: Optional[Path] = typer.Option(
        None,
        "--model-file",
        "-m",
        help="Path to trained model. If omitted, uses default for model type.",
        show_default=False,
    ),
    model_type: str = typer.Option(
        "rf",  # Default to RandomForest
        "--model-type",
        "-t",
        help="Type of model: 'rf' (RandomForest) or 'cnn' (1D-CNN).",
        case_sensitive=False,
    ),
    window_size: int = typer.Option(
        10,
        "--window-size",
        "-w",
        help="Samples per prediction window. MUST match training.",
    ),
):
    """
    Simulates sensor stream, predicts activity (throwing, drinking, driving).
    """
    print("📊 [bold]Starting Real-Time Activity Prediction[/bold]")

    # --- NEW: Logic for random demo data file ---
    if data_file is None:
        print(
            "[bold blue]No data file provided. Running demo with a random file...[/bold blue]"
        )
        data_dir = Path("data")
        demo_files = [
            data_dir / "drinking.csv",
            data_dir / "driving.csv",
            data_dir / "throwing.csv",
        ]
        valid_demo_files = [f for f in demo_files if f.exists()]

        if not valid_demo_files:
            print(
                "[bold red]Error: No demo data files (drinking.csv, etc.) found in 'data/' directory.[/bold red]"
            )
            raise typer.Exit(code=1)

        data_file = random.choice(valid_demo_files)
        print(f"🎲 Randomly selected demo file: [green]{data_file.name}[/green]")

    # --- NEW: Manual validation for data_file ---
    if not data_file.exists():
        print(f"[bold red]Error: Data file not found at {data_file}[/bold red]")
        raise typer.Exit(code=1)

    # --- NEW: Logic for automatic model file selection ---
    if model_file is None:
        if model_type.lower() == "cnn":
            model_file = Path("models/cnn_model.pth")
        elif model_type.lower() == "rf":
            model_file = Path("models/activity_classifier.joblib")
        else:
            print(
                f"[bold red]Error: Unknown model type '{model_type}'. Use 'rf' or 'cnn'.[/bold red]"
            )
            raise typer.Exit(code=1)

    # --- NEW: Manual validation for model_file ---
    if not model_file.exists():
        print(
            f"[bold red]Error: Model file not found at {model_file}. "
            f"Run 'train' or 'train-dl' first.[/bold red]"
        )
        raise typer.Exit(code=1)

    print(
        f"🧠 Loading [bold]{model_type.upper()}[/bold] model from: "
        f"[cyan]{model_file}[/cyan]"
    )

    try:
        predictor = ActivityPredictor(model_path=model_file, model_type=model_type)
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

    try:
        while True:
            sample = driver.read()
            if sample is None:
                print("\n🏁 [yellow]End of sensor stream.[/yellow]")
                break

            window.append(sample)

            if len(window) == window_size:
                try:
                    prediction_label = predictor.predict(window)

                    if prediction_label == "throwing":
                        formatted_label = (
                            f"[bold yellow]{prediction_label}[/bold yellow]"
                        )
                    elif prediction_label == "drinking":
                        formatted_label = f"[bold cyan]{prediction_label}[/bold cyan]"
                    elif prediction_label == "driving":
                        formatted_label = (
                            f"[bold magenta]{prediction_label}[/bold magenta]"
                        )
                    else:  # Fallback for "Unknown"
                        formatted_label = f"[bold red]{prediction_label}[/bold red]"

                    print(f"Prediction: {formatted_label}")

                except Exception as e:
                    print(
                        f"[bold red]Error during prediction calculation: {e}[/bold red]"
                    )

                window = []  # Clear window for next batch

    except KeyboardInterrupt:
        print("\n🛑 [yellow]Stream stopped by user.[/yellow]")
    except Exception as e:
        print(f"\n[bold red]An error occurred during the stream: {e}[/bold red]")
        raise typer.Exit(code=1)


# --- Ensure this block is present at the end ---
if __name__ == "__main__":
    app()
# --- End of file ---

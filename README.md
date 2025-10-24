Here is the complete, final `README.md` file in a single block for you to copy and paste.

````markdown
# Real-Time Multi-Class Activity Recognition (Classical ML vs. 1D-CNN)

![Live Demo](docs/demo.gif)

## 🚀 Project Overview

This project is an end-to-end machine learning application that classifies human activities (throwing, drinking, driving) from sensor data in real-time.

It features a **unified inference engine** that can run either a classical **RandomForest** model or a modern **1D-Convolutional Neural Network (CNN)**, allowing for direct performance comparison. The entire application is built with a modular `src` layout, automatically tested with `pre-commit` hooks and GitLab CI/CD, and is fully containerized with Docker.

## ✨ Features

* **Multi-Model Engine:** Run real-time predictions using either a `RandomForest` or a `1D-CNN` model via a simple CLI flag (`--model-type`).
* **Deep Learning Pipeline:** Includes a full PyTorch pipeline for training a 1D-CNN on raw sensor sequences.
* **Classical ML Pipeline:** Includes a full Scikit-learn pipeline with automated feature extraction (`mean`, `std`, `rms`).
* **Command-Line Interface (CLI):** Provides user-friendly commands (`train`, `train-dl`, `predict-stream`) built with Typer and Rich.
* **Containerized:** A `Dockerfile` is included for building and running the application in any reproducible environment.
* **Automated Workflow:**
    * **GitLab CI/CD:** Automatically runs linting (`flake8`) and unit tests (`pytest`) on every push to ensure code quality.
    * **Pre-commit Hooks:** Automatically formats code (`black`), lints (`flake8`), and runs tests (`pytest`) before each commit.
* **Dependency Management:** Uses Poetry for robust dependency management.

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **ML / DL:** Scikit-learn, **PyTorch**, Pandas, NumPy
* **CLI:** Typer, Rich
* **DevOps:** **Docker**, Poetry, Pre-commit, GitLab CI
* **Testing:** Pytest, Pytest-Mock
* **Code Quality:** Black, Flake8
* **Visualization:** Matplotlib

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SoroushHaghi/Job-Prep2025.git](https://github.com/SoroushHaghi/Job-Prep2025.git)
    cd Job-Prep2025
    ```

2.  **Install Poetry:** (If you don't have it installed)
    Follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    Poetry will automatically create a virtual environment (`.venv`) and install all required packages from `pyproject.toml`.
    ```bash
    poetry install
    ```

4.  **Activate pre-commit hooks:**
    This installs the git hooks defined in `.pre-commit-config.yaml`.
    ```bash
    poetry run pre-commit install
    ```

## ▶️ Usage

All commands are run using the `poetry run` prefix to execute them inside the project's virtual environment.

### 1. Build Feature Dataset (for RandomForest)

This step is only required for the classical ML model.
```bash
poetry run python -m src.main build-features
````

### 2\. Train the Models

You can train either or both models.

  * **Train the RandomForest model:**
    (Creates `models/activity_classifier.joblib`)
    ```bash
    poetry run python -m src.main train
    ```
  * **Train the 1D-CNN model:**
    (Creates `models/cnn_model.pth`)
    ```bash
    poetry run python -m src.main train-dl
    ```

### 3\. Run Real-Time Inference Simulation

You can simulate a real-time stream using any of the raw data files (`data/throwing.csv`, `data/drinking.csv`, etc.).

  * **Run with RandomForest (Default):**

    ```bash
    poetry run python -m src.main predict-stream --data-file data/throwing.csv
    ```

  * **Run with 1D-CNN:**

    ```bash
    poetry run python -m src.main predict-stream --data-file data/throwing.csv --model-type cnn --model-file models/cnn_model.pth
    ```

### 4\. Run with Docker

You can also run the inference engine directly within a Docker container.

1.  **Build the Docker image:**

    ```bash
    docker build -t activity-recognition .
    ```

2.  **Run inference using the container:**

    ```bash
    # Run using the RandomForest model (default)
    docker run --rm activity-recognition predict-stream --data-file data/drinking.csv

    # Run using the 1D-CNN model
    docker run --rm activity-recognition predict-stream --data-file data/drinking.csv --model-type cnn --model-file models/cnn_model.pth
    ```

## 📊 Model Performance & Comparison

Both a classical machine learning model and a modern deep learning model were trained on the same dataset. The 1D-CNN showed a clear performance advantage based on the latest training run.

| Metric | RandomForest (Classical ML) | 1D-CNN (Deep Learning) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 91.7% | **93.0%** |
| Throwing (F1-Score) | 0.94 | **0.96** |
| Drinking (F1-Score) | 0.89 | **0.90** |
| Driving (F1-Score) | 0.92 | 0.92 |

**RandomForest Confusion Matrix:**

## 📁 Project Structure

```
├── data/               # Raw data files (throwing.csv, etc.) and generated features.csv
├── docs/               # Documentation files (confusion_matrix.png, demo.gif)
├── models/             # Saved trained models (.joblib and .pth)
├── src/                # Main source code
│   ├── __init__.py
│   ├── app.py            # Core processing logic, constants (legacy)
│   ├── dataset_builder.py # Script to build features.csv
│   ├── drivers.py        # Sensor driver classes (SimulationDriver)
│   ├── features.py       # Feature calculation functions
│   ├── main.py           # CLI definition using Typer
│   ├── model.py          # (NEW) 1D-CNN PyTorch model definition
│   ├── model_trainer.py  # (RF) Model training and evaluation script
│   ├── predictor.py      # (NEW) Multi-model inference engine
│   ├── train_dl.py       # (NEW) 1D-CNN training and evaluation script
│   └── utils.py          # (NEW) Shared helper functions and constants
├── tests/              # Unit tests
│   ├── __init__.py
│   └── test_*.py        # Pytest files for different modules
├── .gitignore
├── .gitlab-ci.yml      # GitLab CI/CD configuration
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── Dockerfile          # (NEW) Docker container definition
├── poetry.lock
├── pyproject.toml      # Poetry project configuration and dependencies
└── README.md           # This file
```
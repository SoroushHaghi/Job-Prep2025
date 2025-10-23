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

All commands are run from within the Poetry virtual environment.

**First, activate the environment:**
```bash
poetry shell
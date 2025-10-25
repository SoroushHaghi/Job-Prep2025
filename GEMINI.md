# GEMINI.md

## Project Overview

This is a Python-based machine learning project for real-time activity recognition. It classifies human actions (throwing, drinking, driving) from sensor data streams using two different models: a classical Random Forest model and a 1D-Convolutional Neural Network (1D-CNN).

The project is structured as a command-line application using Typer and Rich. It includes a complete workflow for data processing, model training, and real-time inference simulation.

**Core Technologies:**

*   **Machine Learning:** Scikit-learn, PyTorch, Pandas, NumPy
*   **CLI:** Typer, Rich
*   **DevOps:** Docker, Poetry, Pre-commit, GitLab CI
*   **Testing & Quality:** Pytest, Black, Flake8

## Building and Running

1.  **Install Dependencies:**
    ```bash
    poetry install
    ```

2.  **Build Feature Dataset (for RandomForest):**
    ```bash
    poetry run python -m src.main build-features
    ```

3.  **Train Models:**
    *   **RandomForest:**
        ```bash
        poetry run python -m src.main train
        ```
    *   **1D-CNN:**
        ```bash
        poetry run python -m src.main train-dl
        ```

4.  **Run Real-Time Inference Simulation:**
    *   **RandomForest:**
        ```bash
        poetry run python -m src.main predict-stream --data-file data/throwing.csv
        ```
    *   **1D-CNN:**
        ```bash
        poetry run python -m src.main predict-stream --data-file data/throwing.csv --model-type cnn --model-file models/cnn_model.pth
        ```

5.  **Run Tests:**
    ```bash
    poetry run pytest
    ```

## Development Conventions

*   **Dependency Management:** Project dependencies are managed with Poetry.
*   **Code Formatting:** Code is formatted with Black, enforced by a pre-commit hook.
*   **Linting:** Flake8 is used for linting, enforced by a pre-commit hook and in the CI/CD pipeline.
*   **Testing:** Unit tests are written with Pytest and are located in the `tests/` directory.
*   **CI/CD:** A GitLab CI/CD pipeline is configured to automatically run linting and tests on every commit.
